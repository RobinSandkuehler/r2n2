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

# Loss base class (standard from PyTorch)
class _PairwiseImageLoss(th.nn.modules.Module):
    def __init__(self, fixed_image, moving_image, size_average=True, reduce=True):
        super(_PairwiseImageLoss, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self.name = "parent"

        self._weight = 1
    def set_loss_weight(self, weight):
        self._weight = weight

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return tensor.mean() * self._weight
        if not self._size_average and self._reduce:
            return tensor.sum() * self._weight
        if not self.reduce:
            return tensor * self._weight


class MSE(_PairwiseImageLoss):
    r""" The mean square error loss is a simple and fast to compute point-wise measure
    which is well suited for monomodal image registration.
    .. math::
         \mathcal{S}_{\text{MSE}} := \frac{1}{\vert \mathcal{X} \vert}\sum_{x\in\mathcal{X}}
          \Big(I_M\big(x+f(x)\big) - I_F\big(x\big)\Big)^2
    Args:
        fixed_image (Image): Fixed image for the registration
        moving_image (Image): Moving image for the registration
        size_average (bool): Average loss function
        reduce (bool): Reduce loss function to a single value
    """

    def __init__(self, size_average=True, reduce=True):
        super(MSE, self).__init__(size_average, reduce)

        self.name = "mse"

    def forward(self, displacement, fixed_image, warped_image):

        # print("shape", displacement.shape)

        mask = th.zeros_like(fixed_image, dtype=th.uint8, device=fixed_image.device)
        if displacement.shape[0] > 1:
            for dim in range(displacement.size()[-1]):
                mask += (displacement[..., dim].gt(1)).unsqueeze(1) + (displacement[..., dim].lt(-1)).unsqueeze(1)
        else:
            for dim in range(displacement.size()[-1]):
                mask += (displacement[..., dim].gt(1)) + (displacement[..., dim].lt(-1))

        mask = mask == 0

        value_image = (warped_image - fixed_image).pow(2)

        value = th.masked_select(value_image, mask)

        return self.return_loss(value), value_image


