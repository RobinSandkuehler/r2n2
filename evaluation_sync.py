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
import numpy as np
import os
import SimpleITK as sitk
import visdom as vis
import loss.image as il
from utils.transformation import compute_grid


def get_fixe_image_filename(path, filenames):

    image_mean = th.zeros(len(filenames), dtype=th.float32)

    for idx, filename in enumerate(filenames):
        image = sitk.ReadImage(os.path.join(path, filename), sitk.sitkFloat32)
        image_mean[idx] = th.mean(th.tensor(sitk.GetArrayFromImage(image)).squeeze())

    mean_sequence = th.mean(image_mean)

    fixed_image_index = th.argmin(th.abs(image_mean - mean_sequence))

    return filenames[fixed_image_index]


class Evaluater():
    def __init__(self, args, image_size, eval_iteration=0):
        self._args = args
        self._image_size = image_size
        self._viz = vis.Visdom(port=args.port)

        self._device = th.device("cuda:" + str(args.gpu_ids[0]))

        self._grid = compute_grid([image_size[0], image_size[1]], device=self._device)

        if args.image_loss_eval == "MSE":
            self._image_loss = il.MSE()
        else:
            print("Image loss is not suported")

        self._eval_iterations = eval_iteration
        self._min_global_evaluation_error = 1e10

        th.manual_seed(args.seed)

        self._patients = sorted(os.listdir(args.eval_path))

        # compute mean image of all data
        self._mean_image_filenames = []

        for patient in self._patients:
            examinations = sorted(os.listdir(os.path.join(args.eval_path, patient)))
            for exa in examinations:
                slices = sorted(os.listdir(os.path.join(args.eval_path, patient, exa)))
                for image_slice in slices:
                    slice_path = os.path.join(args.eval_path, patient, exa, image_slice)
                    images = sorted(os.listdir(slice_path))
                    self._mean_image_filenames.append(get_fixe_image_filename(slice_path, images))

        self._out_path_image_data = os.path.join(args.o, "image_data")
        if not os.path.exists(self._out_path_image_data):
            os.makedirs(self._out_path_image_data)


        self._win = None

    @property
    def eval_iterations(self):
        return self._eval_iterations

    def evaluation(self, model):

        model.eval()

        slice_index_global = 0
        gloabl_eval_error = []

        for patient in self._patients:
            examinations = sorted(os.listdir(os.path.join(self._args.eval_path, patient)))
            image_loss_examination = 0
            for exa in examinations:
                slices = sorted(os.listdir(os.path.join(self._args.eval_path, patient, exa)))
                image_loss_slices = 0
                for image_slice in slices:
                    slice_path = os.path.join(self._args.eval_path, patient, exa,  image_slice)
                    image_filenames = sorted(os.listdir(slice_path))

                    output_path = os.path.join(self._out_path_image_data, patient, exa, image_slice)

                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    fix_image_filename = os.path.join(slice_path, self._mean_image_filenames[slice_index_global])

                    fixed_image = sitk.ReadImage(os.path.join(slice_path, fix_image_filename), sitk.sitkFloat32)

                    fixed_image = th.tensor(sitk.GetArrayFromImage(fixed_image)).squeeze().unsqueeze_(0).unsqueeze_(0)
                    fixed_image = fixed_image[:, :, ::1, ::1]
                    fixed_image = fixed_image.to(device=self._device)
                    fixed_image = fixed_image - th.mean(fixed_image)
                    fixed_image = fixed_image / th.std(fixed_image)
                    fixed_image.clamp_(-2, 2)

                    sitk.WriteImage(sitk.GetImageFromArray(fixed_image.detach().cpu().squeeze().numpy()),
                                    os.path.join(output_path, "fixed_"
                                                 + self._mean_image_filenames[slice_index_global][:-4] + ".vtk"))

                    image_loss_images = 0
                    for image_filename in image_filenames:

                        moving_image = sitk.ReadImage(os.path.join(slice_path, image_filename), sitk.sitkFloat32)
                        moving_image = th.tensor(sitk.GetArrayFromImage(moving_image)).squeeze().unsqueeze_(0)\
                            .unsqueeze_(0)
                        moving_image = moving_image[:, :, ::1, ::1]
                        moving_image = moving_image - th.mean(moving_image)
                        moving_image = moving_image / th.std(moving_image)
                        moving_image.clamp_(-2, 2)

                        moving_image = moving_image.to(device=self._device)

                        warped_image = moving_image

                        displacement = th.zeros(1, 2, self._image_size[0], self._image_size[1], device=self._device,
                                                dtype=fixed_image.dtype)

                        model.reset()

                        with th.no_grad():
                            for i in range(self._args.rnn_iter_eval):

                                net_input = th.cat((fixed_image, warped_image.detach()), dim=1)
                                net_ouput = model(net_input)

                                displacement = displacement + net_ouput[0]

                                displacement_trans = displacement.transpose(1, 2).transpose(2, 3) + self._grid
                                warped_image = F.grid_sample(moving_image, displacement_trans)

                            loss, _ = self._image_loss(displacement_trans, fixed_image, warped_image)

                            image_loss_images += loss.data.item()

                        sitk.WriteImage(sitk.GetImageFromArray(warped_image.detach().cpu().squeeze().numpy()),
                                        os.path.join(output_path, "warped_" + image_filename[:-4] + ".vtk"))

                        sitk.WriteImage(sitk.GetImageFromArray(moving_image.detach().cpu().squeeze().numpy()),
                                        os.path.join(output_path, "moving_" + image_filename[:-4] + ".vtk"))

                        sitk.WriteImage(sitk.GetImageFromArray(
                                        displacement.detach().transpose(1, 2).transpose(2, 3).cpu().squeeze().numpy(),
                                        isVector=True),
                                        os.path.join(output_path, "displacement_" + image_filename[:-4] + ".vtk"))

                    slice_index_global += 1

                    image_loss_images /= len(image_filenames)

                    image_loss_slices += image_loss_images

                image_loss_slices /= len(slices)

                image_loss_examination += image_loss_slices

            image_loss_examination /= len(examinations)
            gloabl_eval_error.append(image_loss_examination)

            state = {
                'eval_counter': self._eval_iterations,
                'args': self._args,
                'model': model.state_dict()
            }

            path = os.path.join(self._args.o, "model_last.pt")

            if self._args.gpu_ids[0] >= 0:
                with th.cuda.device(self._args.gpu_ids[0]):
                    th.save(state, path)
            else:
                th.save(state, path)

            if self._min_global_evaluation_error > np.sum(np.array(gloabl_eval_error)):
                self._min_global_evaluation_error = np.sum(np.array(gloabl_eval_error))

                path = os.path.join(self._args.o, "model_best.pt")

                if self._args.gpu_ids[0] >= 0:
                    with th.cuda.device(self._args.gpu_ids[0]):
                        th.save(state, path)
                else:
                    th.save(state, path)

        if self._win is None:
            opts = dict(title="evaluation loss")
            self._win = self._viz.line(X=np.column_stack(np.zeros(len(self._patients))),
                                       Y=np.column_stack(np.array(gloabl_eval_error)), opts=opts)
        else:

            self._viz.line(X=np.column_stack(np.ones(len(self._patients)) * self._eval_iterations),
                           Y=np.column_stack(np.array(gloabl_eval_error)),
                           win=self._win,
                           update='append')

        self._eval_iterations += 1

        model.train()




