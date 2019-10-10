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

from utils.transformation import compute_grid
from model.model_utils import SpatialSoftmax, SpatialSoftmaxPos


class GRUCell(th.nn.Module):
    def __init__(self, input_channels, output_channels, keep_rate_x=1, keep_rate_h=0.5,
                 keep_rate_state_h=0.5, keep_rate_state_x=1, kernel_size=3, instance_norm=False):
        super(GRUCell, self).__init__()

        self._instance_norm = instance_norm

        self._bias_r = th.nn.Parameter(th.Tensor(output_channels))
        self._bias_z = th.nn.Parameter(th.Tensor(output_channels))
        self._bias = th.nn.Parameter(th.Tensor(output_channels))

        self._w_r = th.nn.Parameter(th.Tensor(output_channels, input_channels, kernel_size, kernel_size))
        self._w_z = th.nn.Parameter(th.Tensor(output_channels, input_channels, kernel_size, kernel_size))
        self._w = th.nn.Parameter(th.Tensor(output_channels, input_channels, kernel_size, kernel_size))

        self._u_r = th.nn.Parameter(th.Tensor(output_channels, output_channels, kernel_size, kernel_size))
        self._u_z = th.nn.Parameter(th.Tensor(output_channels, output_channels, kernel_size, kernel_size))
        self._u = th.nn.Parameter(th.Tensor(output_channels, output_channels, kernel_size, kernel_size))

        th.nn.init.xavier_normal_(self._w_r)
        th.nn.init.xavier_normal_(self._w_z)
        th.nn.init.xavier_normal_(self._w)

        th.nn.init.xavier_normal_(self._u_r)
        th.nn.init.xavier_normal_(self._u_z)
        th.nn.init.xavier_normal_(self._u)

        self._bias_r.data.fill_(0)
        self._bias_z.data.fill_(0)
        self._bias.data.fill_(1)

        self._keep_rate_state_h = keep_rate_state_h
        self._keep_rate_state_x = keep_rate_state_x
        self._keep_rate_x = keep_rate_x
        self._keep_rate_h = keep_rate_h

        self._padding = int((kernel_size - 1) / 2)

        self._last_state = None

        if self._keep_rate_x < 1:
            self.register_buffer("_drop_value_w_r", th.zeros_like(self._w_r))
            self.register_buffer("_drop_value_w_z", th.zeros_like(self._w_z))

        if self._keep_rate_h < 1:
            self.register_buffer("_drop_value_u_r", th.zeros_like(self._u_r))
            self.register_buffer("_drop_value_u_z", th.zeros_like(self._u_z))

        if self._keep_rate_state_h < 1 or self._keep_rate_state_x < 1:
            if self._keep_rate_state_x < 1:
                self.register_buffer("_drop_value_w", th.zeros_like(self._w))
            if self._keep_rate_state_h < 1:
                self.register_buffer("_drop_value_u", th.zeros_like(self._u))

        if self._instance_norm:
            self._instance_norm_update = th.nn.InstanceNorm2d(output_channels)
            self._instance_norm_forget = th.nn.InstanceNorm2d(output_channels)
            self._instance_norm_propose = th.nn.InstanceNorm2d(output_channels)

    def reset_state(self):
        self._last_state = None

    def forward(self, x):

        if self._keep_rate_x < 1 and self.training:
            self._drop_value_w_r.data.normal_(1, np.sqrt(1 - self._keep_rate_x) / self._keep_rate_x)
            self._drop_value_w_z.data.normal_(1, np.sqrt(1 - self._keep_rate_x) / self._keep_rate_x)

            w_r = self._w_r * self._drop_value_w_r
            w_z = self._w_z * self._drop_value_w_z
        else:
            w_r = self._w_r
            w_z = self._w_z

        if self._keep_rate_h < 1 and self.training:
            self._drop_value_u_r.data.normal_(1, np.sqrt(1 - self._keep_rate_h) / self._keep_rate_h)
            self._drop_value_u_z.data.normal_(1, np.sqrt(1 - self._keep_rate_h) / self._keep_rate_h)

            u_r = self._u_r * self._drop_value_u_r
            u_z = self._u_z * self._drop_value_u_z
        else:
            u_r = self._u_r
            u_z = self._u_z

        if (self._keep_rate_state_h < 1 or self._keep_rate_state_x < 1) and self.training:
            if self._keep_rate_state_x < 1:
                self._drop_value_w.data.normal_(1, np.sqrt(1 - self._keep_rate_state_x) / self._keep_rate_state_x)
                w = self._w * self._drop_value_w
            else:
                w = self._w
            if self._keep_rate_state_h < 1:
                self._drop_value_u.data.normal_(1, np.sqrt(1 - self._keep_rate_state_h) / self._keep_rate_state_h)
                u = self._u * self._drop_value_u
            else:
                u = self._u
        else:
            u = self._u
            w = self._w

        r_x = th.nn.functional.conv2d(x, w_r, self._bias_r, padding=self._padding)
        z_x = th.nn.functional.conv2d(x, w_z, self._bias_z, padding=self._padding)
        h_cand_x = th.nn.functional.conv2d(x, w, self._bias, padding=self._padding)

        if self._last_state is not None:
            r_h = th.nn.functional.conv2d(self._last_state, u_r, padding=self._padding)
            z_h = th.nn.functional.conv2d(self._last_state, u_z, padding=self._padding)
        else:
            r_h = 0
            z_h = 0

        if self._instance_norm:
            r = th.sigmoid(self._instance_norm_update(r_x + r_h))
            z = th.sigmoid(self._instance_norm_forget(z_x + z_h))
        else:
            r = th.sigmoid(r_x + r_h)
            z = th.sigmoid(z_x + z_h)
        if self._last_state is not None:
            h_cand_h = th.nn.functional.conv2d(r*self._last_state, u, padding=self._padding)

        else:
            h_cand_h = 0
            self._last_state = 0

        if self._instance_norm:
            h_cand = th.tanh(self._instance_norm_propose(h_cand_x + h_cand_h))
        else:
            h_cand = th.tanh(h_cand_x + h_cand_h)

        h = z*self._last_state + (1 - z)*h_cand

        self._last_state = h

        return h

class PosNet(th.nn.Module):
    def __init__(self, channels, image_size):
        super(PosNet, self).__init__()
        self._conv_1_var_measure = th.nn.Conv2d(channels, 1, kernel_size=1)
        self._conv_1_ssoft = th.nn.Conv2d(channels, 1, kernel_size=1)

        th.nn.init.xavier_normal_(self._conv_1_var_measure.weight)
        th.nn.init.xavier_normal_(self._conv_1_ssoft.weight)

        self._ssoft_trust = SpatialSoftmax(1, image_size)
        self._ssoft = SpatialSoftmaxPos(image_size, 1)

    def forward(self, x):
        x_1 = self._conv_1_var_measure(x)
        x_2 = self._conv_1_ssoft(x)

        ssoft_measure = self._ssoft_trust(x_1)

        position, softmax_attention = self._ssoft(x_2)

        belive_position = 2.0 - th.sum(th.abs(ssoft_measure - softmax_attention), dim=2)


        return position, belive_position


class PossNetKL(th.nn.Module):
    def __init__(self, channels, image_size, activation='leaky_relu', sigma=1):
        super(PossNetKL, self).__init__()
        self._sigma = sigma
        # create grid
        self.register_buffer("_grid", compute_grid([int(image_size[0]), int(image_size[1])]))

        self._conv_input_soft_max = th.nn.Conv2d(channels, 1, kernel_size=1)
        th.nn.init.xavier_normal_(self._conv_input_soft_max.weight)

        self._ssoft_pos = SpatialSoftmaxPos(image_size, 1)

    def forward(self, x):

        y_pos = self._conv_input_soft_max(x)

        position, q_x = self._ssoft_pos(y_pos)

        diff_pos_x = self._grid[:, :, :, 0] - position[:, 0]
        diff_pos_y = self._grid[:, :, :, 1] - position[:, 1]

        p_x = th.exp(-(diff_pos_x.pow(2)/(2*self._sigma**2) + diff_pos_y.pow(2)/(2*self._sigma**2)))
        p_x = p_x / th.sum(p_x)


        belive = 1.0 / (th.sum(p_x.view(-1) * th.log(p_x.view(-1) / (q_x + 1e-5) + 1e-5)) + 1e-5)

        return position, belive


class ParameterNet(th.nn.Module):
    def __init__(self, channels, image_size, sigma_max=1, hidden_layers=1024, kernel_size=3, activation='leaky_relu',
                 batch_size=1, p=1):
        super(ParameterNet, self).__init__()

        self._batch_size = batch_size
        self._image_size = image_size
        self._channels = channels
        self._sigma_max = sigma_max

        self.register_buffer("_p", th.tensor(p))

        self._conv_input = th.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1)

        if activation == 'leaky_relu':
            self._activation = th.nn.LeakyReLU()
            th.nn.init.kaiming_normal_(self._conv_input.weight)
        elif activation == 'sigmoid':
            self._activation = th.nn.Sigmoid()
            th.nn.init.xavier_normal_(self._conv_input.weight)
        elif activation == 'tanh':
            self._activation = th.nn.Tanh()
            th.nn.init.xavier_normal_(self._conv_input.weight)

        self._spatial_softmax_content = SpatialSoftmax(int(self._channels / 2), image_size=self._image_size)

        self._conv_content = th.nn.Conv2d(int(self._channels / 2), int(self._channels / 2), kernel_size=3,
                                          padding=1)

        self._fc = th.nn.Linear(int(self._channels/2), hidden_layers)
        self._fc_2 = th.nn.Linear(hidden_layers, 5)

        self._fc_2.weight.data.fill_(0)
        self._fc_2.bias.data.fill_(0)
        self._fc.bias.data.fill_(0)

    def forward(self, x):

        x = self._conv_input(x)

        content_softmax = self._spatial_softmax_content(x[:, int(self._channels / 2):, ...])

        content = self._conv_content(self._activation(x[:, :int(self._channels / 2), ...])).view(self._batch_size,
                                                                                            int(self._channels / 2),
                                                                                            int(self._image_size[0] *
                                                                                            self._image_size[1]))

        content_sum = self._activation((content*content_softmax)).sum(dim=2)

        cp_param = self._fc_2(self._activation(self._fc(content_sum)))

        sigma = F.sigmoid(cp_param[:, :2]).squeeze() * self._sigma_max + 1e-5
        value = th.tanh(cp_param[:, 2:4]).squeeze()
        phi = th.sigmoid(cp_param[:, 4]).squeeze() * np.pi
        p = self._p

        return [sigma, value, phi, p]





class InceptionBlock(th.nn.Module):
    def __init__(self, channels_in, hidden_channel, kernel_size=3, activation='leaky_relu'):

        super(InceptionBlock, self).__init__()

        if activation == 'leaky_relu':
            self._activation = th.nn.LeakyReLU()
        elif activation == 'sigmoid':
            self._activation = th.nn.Sigmoid()
        elif activation == 'tanh':
            self._activation = th.nn.Tanh()

        self._con_1_1_a = th.nn.Sequential(th.nn.Conv2d(channels_in, hidden_channel, kernel_size=1),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation,
                                        th.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[1, kernel_size],
                                                     padding=[0, 1]),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation,
                                        th.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[kernel_size, 1],
                                                     padding=[1, 0]),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation,
                                        th.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[1, kernel_size],
                                                     padding=[0, 1]),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation,
                                        th.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[kernel_size, 1],
                                                     padding=[1, 0]),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation
                                        )

        self._con_1_1_b = th.nn.Sequential(th.nn.Conv2d(channels_in, hidden_channel, kernel_size=1),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation,
                                        th.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[1, kernel_size],
                                                     padding=[0, 1]),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation,
                                        th.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[kernel_size, 1],
                                                     padding=[1, 0]),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation
                                        )

        self._con_1_1_c = th.nn.Sequential(th.nn.Conv2d(channels_in, hidden_channel, kernel_size=1),
                                        th.nn.InstanceNorm2d(hidden_channel),
                                        self._activation)

        self._con_1_1_final = th.nn.Sequential(th.nn.Conv2d(hidden_channel*3, channels_in, kernel_size=1),
                                            th.nn.InstanceNorm2d(channels_in),
                                            self._activation)


    def forward(self, x):

        a = self._con_1_1_a(x)
        b = self._con_1_1_b(x)
        c = self._con_1_1_c(x)

        y = th.cat((a, b, c), dim=1)

        y = self._con_1_1_final(y)

        return self._activation(y + x)


class ResidualBlock(th.nn.Module):
    def __init__(self, channels, kernel_size=3, activation='leaky_relu', instance_norm=False):

        super(ResidualBlock, self).__init__()

        if activation == 'leaky_relu':
            self._activation = th.nn.LeakyReLU()
        elif activation == 'sigmoid':
            self._activation = th.sigmoid
        elif activation == 'tanh':
            self._activation = th.nn.Tanh()

        modules = []

        padding = int((kernel_size - 1) / 2)

        modules.append(th.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=1))
        if instance_norm:
            modules.append(th.nn.InstanceNorm2d(channels))
        modules.append(self._activation)
        modules.append(th.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=1))
        if instance_norm:
            modules.append(th.nn.InstanceNorm2d(channels))

        self._model = th.nn.Sequential(*modules)

        # init all layers
        for layer in self._model.children():
            if isinstance(layer, th.nn.Conv2d):
                if activation == 'leaky_relu':
                    th.nn.init.kaiming_normal_(layer.weight)
                elif activation == 'sigmoid' or activation == 'tanh':
                    th.nn.init.xavier_normal_(layer.weight,  gain=th.nn.init.calculate_gain(activation))
                else:
                    print("Initialisation for given activation not defined")

    def forward(self, x):

        return self._activation(self._model(x) + x)


class StackedResidualBlock(th.nn.Module):
    def __init__(self, channels, layer=3, activation='leaky_relu', kernelsize=3, instance_norm=False):
        super(StackedResidualBlock, self).__init__()

        modules = []

        for _ in range(layer):
            modules.append(ResidualBlock(channels, kernel_size=kernelsize, activation=activation,
                                         instance_norm=instance_norm))

        self._model = th.nn.Sequential(*modules)

    def forward(self, x):

        return self._model(x)


class GRUBlock(th.nn.Module):
    def __init__(self, image_size, input_channels, output_channels, padding=0, is_final_block=False,
                 args=None, gru_kernel_size=3):
        super(GRUBlock, self).__init__()

        self._output_channels = output_channels
        self._input_channels = input_channels
        self._batch_size = args.batch_size
        self._args = args

        self._is_final_block = is_final_block

        if args is not None:
            if args.conv_block_type == "RESI":
                self._conv_block = StackedResidualBlock(input_channels, layer=args.layer_residual,
                                                        activation=args.activation, kernelsize=gru_kernel_size,
                                                        instance_norm=args.use_instance_norm)
            elif args.conv_block_type == "INCEP":
                self._conv_block = InceptionBlock(input_channels, hidden_channel=args.hidden_inception,
                                                  activation=args.activation)
            else:
                print("conv layer network type not kwnon!")
                exit(-1)

            self._image_size = np.array(image_size).astype(dtype=np.int32)
        else:
            print("No arguments passed!")
            exit(-1)

        self._gru = GRUCell(input_channels, input_channels, keep_rate_x=args.keep_rate_gate_x,
                            keep_rate_h=args.keep_rate_gate_h, keep_rate_state_h=args.keep_rate_candidate_h,
                            keep_rate_state_x=args.keep_rate_candidate_x, kernel_size=gru_kernel_size)

        if args.pos_net == "sim":
            self._pos_net = PosNet(input_channels, self._image_size)
        elif args.pos_net == "kl":
            self._pos_net = PossNetKL(input_channels, self._image_size)
        else:
            print("position network type not defined")


        self._image_size_param = np.floor((self._image_size - 3)) + 1

        if not self._is_final_block:
            self._conv_ds = th.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)
            self._image_size = np.floor((self._image_size - 3 + 2 * padding) / 2) + 1
        else:
            self._conv_ds = th.nn.Conv2d(input_channels, output_channels, kernel_size=1)

        self._mask = None

        if args.activation == 'leaky_relu':
            self._activation = th.nn.LeakyReLU()
            if not self._is_final_block:
                th.nn.init.kaiming_normal_(self._conv_ds.weight)
        elif args.activation == 'sigmoid':
            self._activation = th.nn.Sigmoid()
            if not self._is_final_block:
                th.nn.init.xavier_normal_(self._conv_ds.weight, gain=th.nn.init.calculate_gain(args.activation))
        elif args.activation == 'tanh':
            self._activation = th.nn.Tanh()
            if not self._is_final_block:
                th.nn.init.xavier_normal_(self._conv_ds.weight, gain=th.nn.init.calculate_gain(args.activation))

        if args.use_instance_norm:
            self._instance_norm = th.nn.InstanceNorm2d(output_channels)

    @property
    def image_size(self):
        return self._image_size

    @property
    def mask(self):
        return self._mask

    def reset_gru(self):
        self._gru.reset_state()

    def forward(self, x):

        x_r = self._conv_block(x)

        if self._args.use_gru_skip:
            out_gru = self._activation(self._gru(x_r) + x_r)
        else:
            out_gru = self._gru(x_r)

        cp_pos, belive_pos = self._pos_net(out_gru)

        if self._args.use_instance_norm:
            out_gru_ds = self._activation(self._instance_norm(self._conv_ds(out_gru)))
        else:
            out_gru_ds = self._activation(self._conv_ds(out_gru))

        return cp_pos, belive_pos, out_gru_ds, out_gru


class GRU_Registration(th.nn.Module):
    def __init__(self, image_size, input_channels, args=None, device='cpu', padding=1):
        super(GRU_Registration, self).__init__()

        self._cp_pos_all = []
        self._cp_value = []
        self._cp_sigma = []
        self._cp_p = []
        self._phi = []

        self._args = args

        hidden_channels = args.channel

        if self._args.use_coord_conv:
            input_channels += 2
            print("Use cord conv")

        self._conv = th.nn.Conv2d(input_channels, hidden_channels[0], kernel_size=7, padding=3, stride=2)

        self._grid = compute_grid(image_size).to(device=device)

        self._sigma_std = args.sigma
        self._batch_size = args.batch_size

        self._input_image_size = image_size

        image_size = np.floor((np.array(image_size) - 7 + 2 * 3) / 2) + 1

        self._hidden_channels = hidden_channels

        self._gru_block_1 = GRUBlock(image_size, hidden_channels[0], hidden_channels[1], padding=padding,
                                     args=args, gru_kernel_size=args.gru_kernel_size[0])

        self._gru_block_2 = GRUBlock(self._gru_block_1.image_size, hidden_channels[1], hidden_channels[2],
                                     padding=padding, args=args, gru_kernel_size=args.gru_kernel_size[1])

        self._gru_block_3 = GRUBlock(self._gru_block_2.image_size, hidden_channels[2], hidden_channels[3],
                                     padding=padding, is_final_block=args.skip_last_down_conv, args=args,
                                     gru_kernel_size=args.gru_kernel_size[2])

        self._param_net = ParameterNet(hidden_channels[-1], self._gru_block_3.image_size, sigma_max=args.sigma,
                                       hidden_layers=args.hidden_layer_param, activation=args.activation, p=args.p)

        if args.activation == 'leaky_relu':
            self._activation = th.nn.LeakyReLU()
            th.nn.init.kaiming_normal_(self._conv.weight, nonlinearity=args.activation)
        elif args.activation == 'sigmoid':
            self._activation = th.nn.Sigmoid()
            th.nn.init.xavier_normal_(self._conv.weight, gain=th.nn.init.calculate_gain(args.activation))
        elif args.activation == 'tanh':
            self._activation = th.nn.Tanh()
            th.nn.init.xavier_normal_(self._conv.weight, gain=th.nn.init.calculate_gain(args.activation))

        if self._args.use_instance_norm:
            self._int_norm = th.nn.InstanceNorm2d(hidden_channels[0])

    def reset(self):
        self._gru_block_1.reset_gru()
        self._gru_block_2.reset_gru()
        self._gru_block_3.reset_gru()

        self._cp_pos_all = []
        self._cp_value = []
        self._cp_sigma = []
        self._cp_p = []
        self._phi = []

    def get_prob_map(self):
        return self._prob_map

    def get_cp_pos(self):
        return self._cp_pos_all

    def get_cp_value(self):
        return self._cp_value

    def get_phi(self):
        return self._phi

    def get_cp_sigma(self):
        return self._cp_sigma

    def get_p(self):
        return self._cp_p

    def forward(self, x):

        if self._args.use_coord_conv:
            x = th.cat((x, self._grid.transpose(3, 2).transpose(2, 1)), dim=1)

        if self._args.use_instance_norm:
            x = self._activation(self._int_norm(self._conv(x)))
        else:
            x = self._activation(self._conv(x))

        cp_pos_1, c_1, x_1, h_1 = self._gru_block_1(x)
        cp_pos_2, c_2, x_2, h_2 = self._gru_block_2(x_1)
        cp_pos_3, c_3, x_3, h_3 = self._gru_block_3(x_2)

        cp_pos = (c_1*cp_pos_1 + c_2*cp_pos_2 + c_3*cp_pos_3)/(c_1 + c_2 + c_3)

        # compute control point parameter
        param = self._param_net(x_3)

        cp_sigma = param[0]
        cp_value = param[1]
        phi = param[2]
        p = param[3]

        if self._args.save_intermediate:
            self._cp_value.append(cp_value.detach()[...])
            self._cp_sigma.append(cp_sigma.detach()[...])
            self._phi.append(phi.detach()[...])
            self._cp_p.append(p.detach()[...])
            self._cp_pos_all.append(cp_pos.detach()[...])

        a = th.cos(phi).pow(2) / (2 * cp_sigma[0].pow(2)) + th.sin(phi).pow(2) / (2 * cp_sigma[1].pow(2))
        b = -th.sin(2 * phi) / (4 * cp_sigma[0].pow(2)) + th.sin(2 * phi) / (4 * cp_sigma[1].pow(2))
        c = th.sin(phi).pow(2) / (2 * cp_sigma[0].pow(2)) + th.cos(phi).pow(2) / (2 * cp_sigma[1].pow(2))

        diff_pos_x = self._grid[:, :, :, 0] - cp_pos[:, 0]
        diff_pos_y = self._grid[:, :, :, 1] - cp_pos[:, 1]

        f_x = th.exp(-(a * diff_pos_x.pow(2) + 2 * b * diff_pos_x * diff_pos_y + c * diff_pos_y.pow(2)).pow(p))
        f_x = f_x.unsqueeze(1)

        displacement = th.cat((f_x * cp_value[0], f_x * cp_value[1]), dim=1)

        return [displacement, f_x, [cp_sigma, cp_value, cp_pos, phi], x_1.detach(), x_2.detach(), x_3.detach()]
