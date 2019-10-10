# Copyright 2019 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division
__author__ = "Robin Sandkuehler"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"


import torch as th
from torch.utils import data
import numpy as np
import visdom as vis
import SimpleITK as sitk
import torch.nn.functional as F
import time
import os


import dataHandler.dataManager as dm
import model.gru_registration as gru
import loss.image as il
import loss.displacement as dl
from utils.transformation import compute_grid
from evaluation_sync import Evaluater
import utils.imageFilter as imfilter


def compute_entropy(f):
    return -th.sum(f * th.log2(f))


def train_sync(args):

    continue_optimization = False
    eval_iteration = 0

    if args.model_state != "":
        state = th.load(args.model_state, map_location='cpu')
        continue_optimization = True
        eval_iteration = state['eval_counter']


    gpu_id = args.gpu_ids[0]

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = th.device("cuda:" + str(gpu_id))

    viz = vis.Visdom(port=args.port)

    data_manager = dm.DataManager(args.training_path, normalize_std=args.normalize_std,
                                  random_sampling=args.random_img_pair)
    image_size = data_manager.image_size()

    # Parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.nb_workers,
              'pin_memory': True}

    training_generator = data.DataLoader(data_manager, **params)

    model = gru.GRU_Registration(image_size, 2, device=device, args=args)

    if continue_optimization:
        model.load_state_dict(state['model'])

    model.train()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters model", params)

    evaluater = Evaluater(args, image_size, eval_iteration=eval_iteration)

    if gpu_id >= 0:
        with th.cuda.device(gpu_id):
            model.cuda()

    if gpu_id >= 0:
        th.cuda.manual_seed(args.seed)
        th.cuda.set_device(gpu_id)

    if args.optimizer == 'RMSprop':
        optimizer = th.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    elif args.optimizer == 'Rprop':
        optimizer = th.optim.Rprop(model.parameters(), lr=args.lr)

    if continue_optimization:
        optimizer.load_state_dict = state['optimizer']

    if args.image_loss == "MSE":
        image_loss = il.MSE()
    else:
        print("Image loss is not suported")

    regulariser = dl.IsotropicTVRegulariser([1.0, 1.0])

    grid = compute_grid([image_size[0], image_size[1]]).cuda()

    train_counter = 0
    if continue_optimization:
        train_counter = state['train_counter']
    loss_plot = None

    print("start optimization")
    scale = 1
    if args.use_diff_loss:
        scale = -1

    while True:

        for fixed_image, moving_image in training_generator:

            fixed_image = fixed_image.cuda()
            moving_image = moving_image.cuda()

            if train_counter % args.eval_interval == 0:
                print("Start evaluation")
                evaluater.evaluation(model)

            image_loss_epoch = 0
            model.reset()
            model.zero_grad()

            warped_image = moving_image

            displacement = th.zeros(args.batch_size, 2, image_size[0], image_size[1], device=fixed_image.device,
                                    dtype=fixed_image.dtype)

            displacement_trans = displacement.transpose(1, 2).transpose(2, 3) + grid

            if args.entropy_regularizer_weight > 0:
                shapes = th.zeros(1, 1, image_size[0], image_size[1], device=fixed_image.device,
                                  dtype=fixed_image.dtype)
                single_entropy = 0

            loss_start, _ = image_loss(displacement_trans, fixed_image, warped_image)

            if args.early_stopping > 0:
                if loss_start.item() < args.early_stopping:
                    continue

            start = time.time()
            for j in range(args.rnn_iter):

                net_input = th.cat((fixed_image, warped_image), dim=1)
                net_ouput = model(net_input)

                displacement = displacement + net_ouput[0]

                if args.entropy_regularizer_weight > 0:
                    f_x = net_ouput[1] / (th.sum(net_ouput[1]) + 1e-5) + 1e-5
                    shapes = shapes + f_x
                    single_entropy = single_entropy + compute_entropy(f_x)

                displacement_trans = displacement.transpose(1, 2).transpose(2, 3) + grid
                warped_image = F.grid_sample(moving_image, displacement_trans)

                loss_, _ = image_loss(displacement_trans, fixed_image, warped_image)


                if args.use_diff_loss:
                    image_loss_epoch = image_loss_epoch + (loss_start - loss_)
                    loss_start = loss_
                else:
                    image_loss_epoch = image_loss_epoch + loss_

                if args.early_stopping > 0:
                    if loss_.item() < args.early_stopping:
                        break

                if args.stop_on_reverse:
                    if loss_.item() <= loss_start.item():
                        loss_start = loss_
                    else:
                        break

            j = j + 1

            displacement_loss = args.reg_weight * regulariser(displacement)

            loss = scale*image_loss_epoch/j + displacement_loss

            if args.entropy_regularizer_weight > 0:
                entropy_loss = (compute_entropy(shapes / j) + single_entropy / j) * args.entropy_regularizer_weight
                loss = loss - entropy_loss
                entropy_loss_value = entropy_loss.data.item()
            else:
                entropy_loss_value = 0

            optimizer.zero_grad()
            loss.backward()

            if args.clip_gradients:
                th.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            end = time.time()

            if train_counter % args.save_model == 0:
                state = {
                    'train_counter': train_counter,
                    'eval_counter':  evaluater.eval_iterations,
                    'args': args,
                    'agent_id': -1,
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict()
                }

                path = os.path.join(args.o, "state_agent_sync.pt")
                th.save(state, path)

            print("iter ", train_counter, "image loss ", image_loss_epoch.item()/j,
                  "displacement loss ", displacement_loss.item(), "loss ", loss.item(), "time", end-start)

            if loss_plot is None:
                opts = dict(title=("loss_value"), width=1000, height=500, showlegend=True)
                loss_value_ = np.column_stack(np.array([image_loss_epoch.data.item() / j, displacement_loss.data.item(),
                                                        entropy_loss_value]))
                loss_plot = viz.line(X=np.column_stack(np.ones(3) * train_counter),
                                     Y=loss_value_, opts=opts)
            else:
                loss_value_ = np.column_stack(np.array([image_loss_epoch.data.item() / j, displacement_loss.data.item(),
                                                        entropy_loss_value]))
                loss_plot = viz.line(X=np.column_stack(np.ones(3) * train_counter),
                                     Y=loss_value_, win=loss_plot, update='append')

            if train_counter % 250 == 0:

                fixed_image_vis = imfilter.normalize_image(fixed_image[0, ...]).cpu().unsqueeze(0)
                moving_image_vis = imfilter.normalize_image(moving_image[0, ...]).cpu().unsqueeze(0)

                displacement_vis = imfilter.normalize_image(displacement[0, ...]).cpu().unsqueeze(0).detach()
                warped_image_vis = imfilter.normalize_image(warped_image[0, ...]).cpu().unsqueeze(0).detach()

                checkerboard_image = sitk.GetArrayFromImage(
                    sitk.CheckerBoard(sitk.GetImageFromArray(moving_image_vis.squeeze().numpy()),
                                      sitk.GetImageFromArray(fixed_image_vis.squeeze().numpy()),
                                      [20, 20]))
                checkerboard_image_vis_nor_reg = th.Tensor(checkerboard_image).unsqueeze(0).unsqueeze(0)

                checkerboard_image = sitk.GetArrayFromImage(
                    sitk.CheckerBoard(sitk.GetImageFromArray(warped_image_vis.squeeze().numpy()),
                                      sitk.GetImageFromArray(fixed_image_vis.squeeze().numpy()),
                                      [20, 20]))
                checkerboard_image_vis = th.Tensor(checkerboard_image).unsqueeze(0).unsqueeze(0)

                image_stack = th.cat((fixed_image_vis, moving_image_vis, displacement_vis[:, 0, ...].unsqueeze(1),
                                      displacement_vis[:, 1, ...].unsqueeze(1), warped_image_vis, checkerboard_image_vis,
                                      checkerboard_image_vis_nor_reg), dim=0)

                opts = dict(title="results")
                viz.images(image_stack, opts=opts,  win=2)

            train_counter += 1

