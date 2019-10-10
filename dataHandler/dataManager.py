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

import os
import torch as th
from torch.utils import data
import SimpleITK as sitk
import numpy as np

import utils.utils as utils
from dataHandler.dataSet import DataSet


class DataManager(data.Dataset):
    def __init__(self, path, normalize_std=True, random_sampling=False):
        super(DataManager, self).__init__()

        self._training_set = []
        self._normalize_std = normalize_std
        self._path = path
        self._random_sampling = random_sampling

        patients = sorted(os.listdir(self._path))

        print("Number of patients in the training set", len(patients))

        if self._random_sampling:
            print("Use random sampling")
            for idx, patient in enumerate(patients):
                folder_name = sorted(os.listdir(os.path.join(self._path, patient)))

                self._training_set.append(DataSet(folder_name, os.path.join(path, patient)))
        else:
            print("Use mean image as fixed image")
            self._data_indices = []

            print("Start indexing data")
            for idx_patient, patient in enumerate(patients):
                examinations = sorted(os.listdir(os.path.join(self._path, patient)))
                print("Done ", (idx_patient/float(len(patients)))*100)
                for idx_examination, examination in enumerate(examinations):
                    slices = sorted(os.listdir(os.path.join(self._path, patient, examination)))
                    for slice_index, slice in enumerate(slices):

                        image_file_names = sorted(os.listdir(os.path.join(self._path, patient, examination, slice)))

                        image_file_names = [f for f in image_file_names if os.path.isfile(os.path.join(self._path, patient, examination, slice, f))
                                            and f.endswith(".dcm")]

                        fixed_image_name = utils.get_fix_image_filename(os.path.join(self._path, patient, examination, slice), image_file_names)

                        for image_file_name in image_file_names:
                            data_tuple = (patient, examination, slice, fixed_image_name, image_file_name)
                            self._data_indices.append(data_tuple)

    def image_size(self):
        fixed_image, moving_image = self.__getitem__(0)
        fixed_image = fixed_image.squeeze()

        return (fixed_image.shape[0], fixed_image.shape[1])

    def __len__(self):
        if self._random_sampling:
            return 123456789
        else:
            return len(self._data_indices)

    def __getitem__(self, index):

        if self._random_sampling:
            np.random.seed(index)
            training_idx = np.random.randint(0, len(self._training_set))
            fixed_image, moving_image = self._training_set[training_idx].sample()
        else:

            data_tuple = self._data_indices[index]

            fixed_image = sitk.ReadImage(os.path.join(self._path, data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3]), sitk.sitkFloat32)
            moving_image = sitk.ReadImage(os.path.join(self._path, data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[4]), sitk.sitkFloat32)

            fixed_image = th.tensor(sitk.GetArrayFromImage(fixed_image))
            moving_image = th.tensor(sitk.GetArrayFromImage(moving_image))

        if self._normalize_std:
            fixed_image = fixed_image - th.mean(fixed_image)
            fixed_image = fixed_image / th.std(fixed_image)
            fixed_image = fixed_image.clamp(-2, 2)

            moving_image = moving_image - th.mean(moving_image)
            moving_image = moving_image / th.std(moving_image)
            moving_image = moving_image.clamp(-2, 2)

        return fixed_image, moving_image








