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
import SimpleITK as sitk


def get_fix_image_filename(path, filenames):

    image_mean = th.zeros(len(filenames), dtype=th.float32)

    for idx, filename in enumerate(filenames):
        image = sitk.ReadImage(os.path.join(path, filename), sitk.sitkFloat32)
        image_mean[idx] = th.mean(th.tensor(sitk.GetArrayFromImage(image)).squeeze())

    mean_sequence = th.mean(image_mean)

    fixed_image_index = th.argmin(th.abs(image_mean - mean_sequence))

    return filenames[fixed_image_index]






