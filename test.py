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
import time
import model.gru_registration as gru
import os
import csv
import SimpleITK as sitk
import loss.image as il
from utils.transformation import compute_grid, Points

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--test-path', default='', help='path of the test data')
parser.add_argument('--state-path', default='', help='')
parser.add_argument('--o', default='', help='output path')

parser.add_argument(
    '--gpu-id',
    type=int,
    default=0
)


def eval_rnn(args, model, fixed_image, moving_image, image_loss, grid):

    model.reset()
    warped_image = moving_image

    image_size = fixed_image.size()

    displacement = th.zeros(1, 2, image_size[-2], image_size[-1], device=fixed_image.device,
                            dtype=fixed_image.dtype)

    image_loss_image = 0

    with th.no_grad():
        for i in range(args.rnn_iter_eval):
            net_input = th.cat((fixed_image, warped_image), dim=1)
            net_output = model(net_input)

            displacement = displacement + net_output[0]

            displacement_trans = displacement.transpose(1, 2).transpose(2, 3) + grid
            warped_image = F.grid_sample(moving_image, displacement_trans)

        loss, _ = image_loss(displacement_trans, fixed_image, warped_image)

        image_loss_image += loss.data.item()

    return image_loss_image, warped_image, displacement


def eval_feed_forward(args, model, fixed_image, moving_image, image_loss, grid):

    with th.no_grad():
        net_input = th.cat((fixed_image, moving_image), dim=1)
        net_output = model(net_input)

        displacement = net_output[0]

        displacement_trans = displacement.transpose(1, 2).transpose(2, 3) + grid
        warped_image = F.grid_sample(moving_image, displacement_trans)

        loss, _ = image_loss(displacement_trans, fixed_image, warped_image)

    return loss.data.item(), warped_image, displacement


def get_fixe_image_filename(path, filenames):

    image_mean = th.zeros(len(filenames), dtype=th.float32)

    for idx, filename in enumerate(filenames):
        image = sitk.ReadImage(os.path.join(path, filename), sitk.sitkFloat32)
        image_mean[idx] = th.mean(th.tensor(sitk.GetArrayFromImage(image)).squeeze())

    mean_sequence = th.mean(image_mean)

    fixed_image_index = th.argmin(th.abs(image_mean - mean_sequence))

    return filenames[fixed_image_index]


def test(args, state,  image_size=[256, 256]):

    args_state = state['args']

    gpu_id = args.gpu_id

    device = th.device("cuda:" + str(gpu_id))

    if gpu_id >= 0:
        th.cuda.set_device(gpu_id)

    patients = sorted(os.listdir(args.test_path))

    # compute mean image of all data
    mean_image_filenames = []

    for patient in patients:
        examinations = sorted(os.listdir(os.path.join(args.test_path, patient)))
        for exa in examinations:
            slices = sorted(os.listdir(os.path.join(args.test_path, patient, exa)))
            for image_slice in slices:
                slice_path = os.path.join(args.test_path, patient, exa, image_slice)
                images = sorted(os.listdir(slice_path))

                images = [f for f in images if os.path.isfile(os.path.join(os.path.join(args.test_path, patient, exa, image_slice, f)))]

                mean_image_filenames.append(get_fixe_image_filename(slice_path, images))

    print(len(mean_image_filenames))
    print(mean_image_filenames)

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

    if args_state.model == "R2NN":
        evaluate_net = eval_rnn
    elif args_state.model == "UNET":
        evaluate_net = eval_feed_forward


    if not os.path.exists(args.o):
        os.makedirs(args.o)


    out_path_image_data = os.path.join(args.o, "image_data")
    if not os.path.exists(out_path_image_data):
        os.makedirs(out_path_image_data)


    slice_index_global = 0
    gloabl_eval_error = []

    for patient in patients:
        if os.path.exists(os.path.join(args.o, "error_" + patient + ".csv")):
            os.remove(os.path.join(args.o, "error_" + patient + ".csv"))

        if os.path.exists(os.path.join(args.o, "tre_" + patient + ".csv")):
            os.remove(os.path.join(args.o, "tre_" + patient + ".csv"))

    for patient in patients:
        examinations = sorted(os.listdir(os.path.join(args.test_path, patient)))
        image_loss_examination = 0
        for exa in examinations:
            slices = sorted(os.listdir(os.path.join(args.test_path, patient, exa)))
            image_loss_slices = 0
            for image_slice in slices:
                slice_path = os.path.join(args.test_path, patient, exa, image_slice)
                image_filenames = sorted(os.listdir(slice_path))
                image_filenames = [f for f in image_filenames if f.endswith(".dcm")]

                output_path = os.path.join(out_path_image_data, patient, exa, image_slice)

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                fix_image_filename = os.path.join(slice_path, mean_image_filenames[slice_index_global])

                fixed_image = sitk.ReadImage(os.path.join(slice_path, fix_image_filename), sitk.sitkFloat32)

                # load fixed image landmarks
                fix_landmarks_filenames = os.path.join(slice_path, "landmarks", "landmarks_" + mean_image_filenames[slice_index_global][:-4] + ".vtk")
                fixed_image_points = Points.read(fix_landmarks_filenames)


                fixed_image = th.tensor(sitk.GetArrayFromImage(fixed_image)).squeeze().unsqueeze_(0).unsqueeze_(0)

                fixed_image = fixed_image.to(device=device)
                fixed_image = fixed_image - th.mean(fixed_image)
                fixed_image = fixed_image / th.std(fixed_image)
                fixed_image.clamp_(-2, 2)

                sitk.WriteImage(sitk.GetImageFromArray(fixed_image.detach().cpu().squeeze().numpy()),
                                os.path.join(output_path, "fixed_"
                                             + mean_image_filenames[slice_index_global][:-4] + ".vtk"))
                image_loss_images = 0
                image_loss_images_csv = []

                tre_slice = []
                tre_slice.append(image_slice)
                tre_slice.append(fix_image_filename)

                image_loss_images_csv.append(image_slice)
                image_loss_images_csv.append(fix_image_filename)

                for image_filename in image_filenames:
                    moving_image = sitk.ReadImage(os.path.join(slice_path, image_filename), sitk.sitkFloat32)

                    # get image properties
                    image_spacing = [1, 1]
                    image_origin = [0, 0]


                    moving_image = th.tensor(sitk.GetArrayFromImage(moving_image)).squeeze().unsqueeze_(0) \
                        .unsqueeze_(0)

                    moving_image = moving_image.to(device=device)
                    moving_image = moving_image - th.mean(moving_image)
                    moving_image = moving_image / th.std(moving_image)
                    moving_image.clamp_(-2, 2)

                    # load moving image landmarks
                    moving_landmarks_filenames = os.path.join(slice_path, "landmarks",  "landmarks_" + image_filename[:-4] + ".vtk")
                    moving_image_points = Points.read(moving_landmarks_filenames)

                    start = time.time()

                    image_loss_f, warped_image, displacement = evaluate_net(args_state, model, fixed_image,
                                                                            moving_image, image_loss, grid)

                    stop = time.time()

                    displacement = displacement.flip(2)
                    displacement = displacement.transpose(1, 2).transpose(2, 3)
                    displacement = displacement.squeeze().to(dtype=th.float64, device='cpu')
                    # transform to itk displacement
                    for dim in range(displacement.shape[-1]):
                        tmp = float(displacement.shape[-dim - 2] - 1)
                        displacement[..., dim] = float(displacement.shape[-dim - 2] - 1) * displacement[..., dim] / 2.0

                    itk_displacement = sitk.GetImageFromArray(displacement.numpy(), isVector=True)
                    itk_displacement.SetSpacing(image_spacing)
                    itk_displacement.SetOrigin(image_origin)

                    #
                    # displacement_al =  Displacement(displacement, image_size=[256, 256], image_spacing=image_spacing,
                    #                                 image_origin=image_origin)

                    # displacement_al.image = displacement_al.image*image_spacing[0]

                    moving_points_transformed = Points.transform(moving_image_points, itk_displacement)

                    tre = Points.TRE(moving_points_transformed, fixed_image_points)

                    tre_slice.append(tre)

                    print("Time", stop-start)



                    image_loss_images_csv.append(image_loss_f)

                    Points.write(os.path.join(output_path, "warped_points_" + image_filename[:-4] + ".vtk"), moving_points_transformed)
                    Points.write(os.path.join(output_path, "moving_points_" + image_filename[:-4] + ".vtk"), moving_image_points)
                    Points.write(os.path.join(output_path, "fixed_points_" + image_filename[:-4] + ".vtk"), fixed_image_points)

                    image_loss_images += image_loss_f

                    sitk.WriteImage(sitk.GetImageFromArray(warped_image.detach().cpu().squeeze().numpy()),
                                    os.path.join(output_path, "warped_" + image_filename[:-4] + ".vtk"))
                    sitk.WriteImage(sitk.GetImageFromArray(moving_image.detach().cpu().squeeze().numpy()),
                                    os.path.join(output_path, "moving_" + image_filename[:-4] + ".vtk"))

                    sitk.WriteImage(sitk.GetImageFromArray(
                        displacement.detach().cpu().squeeze().numpy(), isVector=True),
                                    os.path.join(output_path, "displacement_" + image_filename[:-4] + ".vtk"))

                slice_index_global += 1

                with open(os.path.join(args.o, "error_" + patient + ".csv"), 'a') as csvFile:
                    writer = csv.writer(csvFile, delimiter=',')
                    writer.writerow(image_loss_images_csv)

                with open(os.path.join(args.o, "tre_" + patient + ".csv"), 'a') as csvFile:
                    writer = csv.writer(csvFile, delimiter=',')
                    writer.writerow(tre_slice)

                image_loss_images /= len(image_filenames)

                image_loss_slices += image_loss_images

            image_loss_slices /= len(slices)

            image_loss_examination += image_loss_slices

        image_loss_examination /= len(examinations)
        gloabl_eval_error.append(image_loss_examination)

        with open(os.path.join(args.o, "error_all_patients.csv"), 'a') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow([patient, examinations, image_loss_examination])

if __name__ == "__main__":
    args = parser.parse_args()

    state = th.load(args.state_path, map_location='cpu')



    test(args, state)