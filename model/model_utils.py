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
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SpatialSoftmax(th.nn.Module):
    def __init__(self, channel, image_size, temperature=None):
        super(SpatialSoftmax, self).__init__()
        self.channel = channel
        self._image_size = image_size

        self.softmax_attention = None

        if temperature:
            self.temperature = Parameter(th.ones(1)*temperature)
        else:
            self.temperature = 1.

    def forward(self, feature):

        feature = feature.view(-1, self.channel, int(self._image_size[0]*self._image_size[1]))

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)

        return softmax_attention

class SpatialSoftmaxPos(th.nn.Module):
    def __init__(self, image_size, channel, temperature=None, batch_size=1):
        super(SpatialSoftmaxPos, self).__init__()
        self.height = image_size[0]
        self.width = image_size[1]
        self.channel = channel
        self._batch_size = batch_size

        if temperature:
            self.temperature = Parameter(th.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = th.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = th.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):

        feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)

        expected_x = th.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = th.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = th.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints, softmax_attention





