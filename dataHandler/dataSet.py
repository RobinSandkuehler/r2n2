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
import re
import torch as th
import SimpleITK as sitk
import numpy as np


class Examination():
    def __init__(self, path, name):

        valid = re.compile(r"[0-9]{1,2}_F")

        self._path = os.path.join(path, name)

        self._slices = sorted([f for f in os.listdir(self._path) if valid.match(f)])

    def _get_number_of_images_per_slice(self):

        nb_images = []

        for image_slices in self._slices:
            image_file_names = sorted(os.listdir(os.path.join(self._path, image_slices)))
            image_file_names = [f for f in image_file_names if os.path.isfile(os.path.join(self._path, image_slices, f))]

            nb_images.append(len(image_file_names))

        return nb_images

    def sample(self):

        slice_idx = np.random.randint(0, len(self._slices))

        image_file_names = sorted(os.listdir(os.path.join(self._path, self._slices[slice_idx])))

        image_file_names = [f for f in image_file_names if os.path.isfile(os.path.join(
                            self._path, self._slices[slice_idx], f)) and f.endswith(".dcm")]

        file_names = np.random.choice(image_file_names, 2, replace=False)

        fixed_image = sitk.ReadImage(os.path.join(self._path, self._slices[slice_idx], file_names[0]),
                                     sitk.sitkFloat32)

        moving_image = sitk.ReadImage(os.path.join(self._path, self._slices[slice_idx], file_names[1]),
                                      sitk.sitkFloat32)

        fixed_image = th.tensor(sitk.GetArrayFromImage(fixed_image)).squeeze().unsqueeze_(0)
        moving_image = th.tensor(sitk.GetArrayFromImage(moving_image)).squeeze().unsqueeze_(0)

        return fixed_image, moving_image


class DataSet():
    def __init__(self, examinations, path):

        self._examinations = []
        self._path = path

        for examination in examinations:
            self._examinations.append(Examination(path, examination))
            print("   ", examination)

    def sample(self):
        examination_idx = np.random.randint(0, len(self._examinations))

        fixed_image, moving_image = self._examinations[examination_idx].sample()

        return fixed_image, moving_image
