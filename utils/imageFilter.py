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
import multiprocessing as mp
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(mp.cpu_count())

import torch as th


def normalize_image(image, new_min=0, new_max=255):

    min = th.min(image)
    max = th.max(image)

    if not (max == min):
        image = (image - min)*((new_max - new_min)/(max - min)) + new_min

    return image