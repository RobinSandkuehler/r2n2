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
import sys
import torch as th

os.environ["OMP_NUM_THREADS"] = "1"

from train_sync import train_sync
import parameter as parameter

if __name__ == "__main__":

    args = parameter.parser.parse_args()

    print(args)

    th.manual_seed(args.seed)

    if args.workers == -1:

        out_path_image_data = os.path.join(args.o)
        if not os.path.exists(out_path_image_data):
            os.makedirs(out_path_image_data)
        else:
            args.model_state = os.path.join(args.o, "state_agent_sync.pt")

        sys.stdout = open(os.path.join(args.o, "commandline_args.txt"), 'w')
        print(args)
        sys.stdout = sys.__stdout__

        train_sync(args)

    else:
        print("Multiple worker learning is not implemented")
        exit(-1)
