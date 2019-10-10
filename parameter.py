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


import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--training-path',
    default='',
    help='path of the training data'
)
parser.add_argument(
    '--eval-path',
    default='',
    help='path of the training data'
)
parser.add_argument(
    '--o',
    default='',
    help='path to save the data'
)

parser.add_argument(
    '--model-state',
    default='',
    help='state of the model'
)

parser.add_argument(
    '--gpu-ids',
    type=int,
    default=[0],
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)'
)
parser.add_argument(
    '--workers',
    type=int,
    default=-1,
    metavar='W',
    help='how many training processes to use (default: 32)'
)
parser.add_argument(
    '--nb-workers',
    type=int,
    default=10,
    help='number of worker for the data loader'
)
parser.add_argument(
    '--reg-weight',
    type=float,
    default=1.0,
    help='weight of the regularisation '
)
parser.add_argument(
    '--rnn-iter',
    type=int,
    default=50,
    help='number of training rnn samples'
)
parser.add_argument(
    '--rnn-iter-eval',
    type=int,
    default=50,
    help='number of training rnn samples'
)
parser.add_argument(
    '--image-loss',
    default="MSE",
    help='image loss function'
)
parser.add_argument(
    '--image-loss-eval',
    default="MSE",
    help='image loss function for the evaluation'
)
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)'
)
parser.add_argument(
    '--sigma',
    type=float,
    default=0.25,
    help='maximum sigma for the gaussian kernel'
)

parser.add_argument(
    '--channel',
    type=int,
    default=[32, 64, 128, 256],
    nargs='+',
    help='channels for the gru blocks'
)
parser.add_argument(
    '--port',
    type=int,
    default=8097,
    help='port for the visdom server'
)
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)'
)
parser.add_argument(
    '--keeprate',
    type=float,
    default=0.5,
    help='dropconnet keep rate value'
)

parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop'
)
parser.add_argument(
    '--amsgrad',
    default=True,
    type=lambda x: (str(x).lower()) == 'true',
    metavar='AM',
    help='Adam optimizer amsgrad parameter'
)

parser.add_argument(
    '--batch-size',
    type=int,
    default=1,
    help='batch size for training'
)
parser.add_argument(
    '--use-diff-loss',
    default=False,
    type=lambda x: (str(x).lower()) == 'true',
    help='maximize the difference between loss values'
)
parser.add_argument(
    '--eval-interval',
    type=int,
    default=500,
    help='compute evaluation after n iterations'
)
parser.add_argument(
    '--conv-block-type',
    default="RESI",
    help='network type bevor the GRU'
)
parser.add_argument(
    '--layer-residual',
    type=int,
    default=6,
    help='number of layers of the residual network'
)
parser.add_argument(
    '--hidden-layer-param',
    type=int,
    default=1024,
    help='number of layers iin the parameter network'
)
parser.add_argument(
    '--activation',
    default='leaky_relu',
    help='used activation function'
)
parser.add_argument(
    '--model',
    default='R2NN',
    help='used activation function'
)
parser.add_argument(
    '--nb-preload-image',
    type=int,
    default=32,
    help='number of layers iin the parameter network'
)


parser.add_argument(
    '--keep-rate-gate-x',
    type=float,
    default=1,
    help='keep rate of the gru gate for the input'
)
parser.add_argument(
    '--keep-rate-gate-h',
    type=float,
    default=0.5,
    help='keep rate of the gru for the last state '
)
parser.add_argument(
    '--keep-rate-candidate-x',
    type=float,
    default=1,
    help='keep rate of the gru proposel for the input'
)
parser.add_argument(
    '--keep-rate-candidate-h',
    type=float,
    default=0.5,
    help='keep rate of the gru proposel for the last state'
)
parser.add_argument(
    '--entropy-regularizer-weight',
    type=float,
    default=0,
    help='weight of the entropy regularizer'
)
parser.add_argument(
    '--pos-net',
    default="sim",
    help='pos net type'
)
parser.add_argument(
    '--skip-last-down-conv',
    default=False,
    type=lambda x: (str(x).lower()) == 'true',
    help='pos net type'
)
parser.add_argument(
    '--gru-kernel-size',
    type=int,
    default=[3, 3, 3],
    nargs='+',
    help='channels for the gru blocks'
)

parser.add_argument(
    '--p',
    type=float,
    default=1.0,
    help='scaling generalized gaussian'
)
parser.add_argument(
    '--save-intermediate',
    default=False,
    type=lambda x: (str(x).lower()) == 'true',
    help='save parameter values in all training steps'
)

parser.add_argument(
    '--normalize-std',
    default=True,
    type=lambda x: (str(x).lower()) == 'true',
    help='normalize data'
)

parser.add_argument(
    '--use-instance-norm',
    default=True,
    type=lambda x: (str(x).lower()) == 'true',
    help='normalize data'
)

parser.add_argument(
    '--hidden-inception',
    type=int,
    default=64,
    help='channels for the gru blocks'
)

parser.add_argument(
    '--clip-gradients',
    default=True,
    type=lambda x: (str(x).lower()) == 'true',
    help='clip the gradient'
)

parser.add_argument(
    '--use-gru-skip',
    default=True,
    type=lambda x: (str(x).lower()) == 'true',
    help='clip the gradient'
)

parser.add_argument(
    '--use-coord-conv',
    default=False,
    type=lambda x: (str(x).lower()) == 'true',
    help='add spatial coordinates to the input'
)

parser.add_argument(
    '--save-model',
    type=int,
    default=5000,
    help='number of bins mutual information'
)
parser.add_argument(
    '--random-img-pair',
    default=False,
    type=lambda x: (str(x).lower()) == 'true',
    help='random sampling of an image pair.'
)

parser.add_argument(
    '--early-stopping',
    default=0,
    type=float,
    help='early stopping'
)

parser.add_argument(
    '--stop-on-reverse',
    default=False,
    type=lambda x: (str(x).lower()) == 'true',
    help='stopp if the loss value increase'
)