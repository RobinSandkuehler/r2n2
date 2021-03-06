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
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import model.gru_registration as gru
import os
import SimpleITK as sitk
import loss.image as il
from utils.transformation import compute_grid

import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--f', default='', help='path of the fixed image')
parser.add_argument('--m', default='', help='path of the moving image')
parser.add_argument('--state-path', default='', help='path of the model state')
parser.add_argument('--o', default='', help='output path')

parser.add_argument(
    '--gpu-id',
    type=int,
    default=0
)


def eval_rnn(iterations, model, fixed_image, moving_image, image_loss, grid):

    model.reset()
    warped_image = moving_image

    image_size = fixed_image.size()

    displacement = th.zeros(1, 2, image_size[-2], image_size[-1], device=fixed_image.device,
                            dtype=fixed_image.dtype)

    image_loss_image = 0
    displacement_param = []
    displacement_pixel = []
    single_displacement = []
    warped_local_image = []
    warped_local_image.append(warped_image)
    with th.no_grad():
        for i in range(iterations):

            net_input = th.cat((fixed_image, warped_image), dim=1)
            net_output = model(net_input)

            displacement = displacement + net_output[0]
            displacement_pixel.append(displacement)
            displacement_param.append(net_output[2])
            single_displacement.append(net_output[0])
            print(i, net_output[2])

            displacement_trans = displacement.transpose(1, 2).transpose(2, 3) + grid
            warped_image = F.grid_sample(moving_image, displacement_trans)
            warped_local_image.append(warped_image)

        loss, _ = image_loss(displacement_trans, fixed_image, warped_image)

        image_loss_image += loss.data.item()

    return image_loss_image, warped_image, displacement, displacement_param, displacement_pixel, single_displacement, warped_local_image


def reg(args, state, fix_image_filename, moving_image_filename, iterations):

    args_state = state['args']

    image_size = [256, 256]

    gpu_id = args.gpu_id

    device = th.device("cuda:" + str(gpu_id))

    if gpu_id >= 0:
        th.cuda.set_device(gpu_id)

    if args_state.model == "R2NN":
        model = gru.GRU_Registration(image_size, 2, args=args_state, device=device)
    else:
        raise ValueError('model type {0} is not known'.format(args_state.model))

    model.eval()
    model.load_state_dict(state['model'])
    print("model loaded")

    if gpu_id >= 0:
        with th.cuda.device(gpu_id):
            model.cuda()

    if args_state.image_loss == "MSE":
        image_loss = il.MSE()
    else:
        print("Image loss is not suported")

    grid = compute_grid([image_size[0], image_size[1]], device=device)

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    fixed_image = sitk.ReadImage(fix_image_filename, sitk.sitkFloat32)
    fixed_image = th.tensor(sitk.GetArrayFromImage(fixed_image)).squeeze().unsqueeze_(0).unsqueeze_(0)

    fixed_image = fixed_image.to(device=device)
    fixed_image = fixed_image - th.mean(fixed_image)
    fixed_image = fixed_image / th.std(fixed_image)
    fixed_image.clamp_(-2, 2)

    moving_image = sitk.ReadImage(moving_image_filename, sitk.sitkFloat32)
    moving_image = th.tensor(sitk.GetArrayFromImage(moving_image)).squeeze().unsqueeze_(0).unsqueeze_(0)

    moving_image = moving_image.to(device=device)
    moving_image = moving_image - th.mean(moving_image)
    moving_image = moving_image / th.std(moving_image)
    moving_image.clamp_(-2, 2)

    image_loss_f, warped_image, displacement, displacement_param, displacement_pixel, single_displacement, warped_local_image = eval_rnn(iterations,
                                                                            model,
                                                                            fixed_image,
                                                                            moving_image,
                                                                            image_loss, grid)

    show_text = False

    for idx, param in enumerate(displacement_param):

        sigma = 2 * param[0].squeeze().cpu().numpy() * 255
        pos = ((param[2].squeeze().cpu().numpy() + 1) / 2) * 255
        angle = -(param[3].cpu().numpy() * 180.0) / np.pi

        displacement_sum = th.sqrt(displacement_pixel[idx][0, 0, ...].pow(2) + displacement_pixel[idx][0, 1, ...].pow(2))
        fig = plt.imshow(displacement_sum.cpu().squeeze().numpy(), cmap='jet', vmax=0.08, vmin=0)

        ax = plt.gca()
        ax.add_patch(Ellipse(pos, width=sigma[0], height=sigma[1],
                                 angle=angle,
                                 edgecolor='white',
                                 facecolor='none',
                                 linewidth=2))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        if show_text:
            plt.text(8, 250, r"transformation sum: $t=" + str(idx) + "$", {'color': 'w', 'fontsize': 18})

        plt.savefig(os.path.join(args.o, "displacement_sum_" + str(idx) + ".png"), bbox_inches='tight', pad_inches=0)
        plt.close()

        ##############################################################################################################
        displacement_local = th.sqrt(single_displacement[idx][0, 0, ...].pow(2) + single_displacement[idx][0, 1, ...].pow(2))
        fig = plt.imshow(displacement_local.cpu().squeeze().numpy(), cmap='jet',vmax=0.03, vmin=0)

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        if show_text:
            plt.text(32, 15, r"network output: $t=" + str(idx) + "$", {'color': 'w', 'fontsize': 18})
            plt.text(8, 250, r"transformation local: $t=" + str(idx) + "$", {'color': 'w', 'fontsize': 18})
        # plt.show()
        plt.savefig(os.path.join(args.o, "displacement_local_" + str(idx) + ".png"), bbox_inches='tight', pad_inches=0)
        plt.close()

        ##############################################################################################################
        fig = plt.imshow(warped_local_image[idx].cpu().squeeze().numpy(), cmap='gray')

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.axis('off')
        if show_text:
            if idx == 0:
                plt.text(16, 250, r"moving image", {'color': 'r', 'fontsize': 18})
            else:
                plt.text(16, 250, r"warped image: $t=" + str(idx - 1) + "$", {'color': 'r', 'fontsize': 18})

            plt.text(3, 15, r"input: $t=" + str(idx) + "$", {'color': 'r', 'fontsize': 18})

        plt.savefig(os.path.join(args.o, "warped_loacl_input_" + str(idx) + ".png"), bbox_inches='tight', pad_inches=0)

        plt.close()

        ##############################################################################################################
        fig = plt.imshow(warped_local_image[idx + 1].cpu().squeeze().numpy(), cmap='gray')
        ax = plt.gca()
        ax.add_patch(Ellipse(pos, width=sigma[0], height=sigma[1],
                             angle=angle,
                             edgecolor='white',
                             facecolor='none',
                             linewidth=2))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.axis('off')
        if show_text:
            plt.text(16, 250, r"warped image: $t=" + str(idx) + "$", {'color': 'r', 'fontsize': 18})
        plt.savefig(os.path.join(args.o, "warped_loacl_output_" + str(idx) + ".png"), bbox_inches='tight', pad_inches=0)
        plt.close()

        ##############################################################################################################
        diff_image = warped_local_image[idx].cpu().squeeze().data.fill_(1).numpy()
        fig = plt.imshow(diff_image, cmap='gray')

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        if show_text:
            plt.text(128, 150, "Recurrent Registration\n" "Neural Networks for\n" "Deformable Image\n""Registration",
                    {'color': 'w', 'fontsize': 18}, ha='center', wrap=True, )

        plt.savefig(os.path.join(args.o, "diff_image_" + str(idx) + ".png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    fig = plt.imshow(fixed_image.cpu().squeeze().numpy(), cmap='gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    if show_text:
        plt.text(64, 250, r"fixed image", {'color': 'r', 'fontsize': 18})
        plt.text(180, 15, r"network ", {'color': 'r', 'fontsize': 18})
    plt.savefig(os.path.join(args.o, "fixed_image.png"), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    args = parser.parse_args()

    state = th.load(args.state_path, map_location='cpu')
    reg(args, state, args.f, args.m, 25)