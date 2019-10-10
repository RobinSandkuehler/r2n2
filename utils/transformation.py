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
import SimpleITK as sitk


def compute_grid(image_size, dtype=th.float32, device='cpu'):

    dim = len(image_size)

    if dim == 2:
        nx = image_size[1]
        ny = image_size[0]

        x = th.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = th.linspace(-1, 1, steps=ny).to(dtype=dtype)

        x = x.expand(ny, -1)
        y = y.expand(nx, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return th.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = th.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = th.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = th.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return th.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        print("Error " + dim + "is not a valid grid type")



class Points:
    """
        Class implementing functionality for dealing with points:

        - read/write: supported formats are pts and vtk (polydata)
        - transform: transform the points given a displacement field
        - TRE: calculates the target registration error between two point sets
    """
    @staticmethod
    def read(filename):
        """
        Read points from file. Following formats are supported:

        - pts: each point is represended in one line where the coordinates are separated with a tab

        - vtk: the vtk polydata is supported as well

        filename (str): filename
        return (array): two dimensional array
        """
        if filename.endswith("pts"):
            points = []
            with open(filename) as f:
                lines = f.readlines()
                for l in lines:
                    points.append([float(p) for p in l.split()])
            return np.array(points)

        elif filename.endswith("vtk"):
            with open(filename) as f:
                lines = f.readlines()
                if not lines[1] == "vtk output\n" and \
                    not lines[2] == "ASCII\n" and \
                    not lines[3] == "DATASET POLYDATA\n":
                    raise Exception("Tried to read corrupted vtk polydata file")
                n = int(lines[4].split()[1])

                one_line = '\t'.join(''.join(lines[5:]).split('\n'))
                one_line = [float(p) for p in one_line.split()]
                return np.array(one_line).reshape((n, 3))

        else:
            raise Exception("Format not supported: "+str(filename))

    @staticmethod
    def write(filename, points):
        """
        Write point list to hard drive
        filename (str): destination filename
        points (array): two dimensional array
        """
        if filename.endswith("pts"):
            with open(filename, 'w') as f:
                for p in points:
                    f.write('\t'.join([str(v) for v in p])+'\n')

        elif filename.endswith("vtk"):
            n = points.shape[0]
            with open(filename, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("vtk output\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")
                f.write("POINTS "+str(n)+" float\n")
                for p in points:
                    f.write('\t'.join([str(v) for v in p])+'\n')

        else:
            raise Exception("Format not supported: "+str(filename))

    @staticmethod
    def transform(points, displacement):
        """
        Transforms a set of points with a displacement field

        points (array): array of points
        displacement (SimpleITK.Image | Displacement ): displacement field to transform points
        return (array): transformed points
        """
        if type(displacement) == sitk.SimpleITK.Image:
            df_transform = sitk.DisplacementFieldTransform(displacement)
        else:
            raise Exception("Datatype of displacement field not supported.")

        df_transform.SetSmoothingOff()

        transformed_points = np.zeros_like(points)
        for i in range(points.shape[0]):
            tmp = df_transform.TransformPoint([points[i, 0], points[i, 1]])
            transformed_points[i, :] = [tmp[0], tmp[1], 0]

        return transformed_points

    @staticmethod
    def TRE(points1, points2):
        """
        Computes the average distance between points in points1 and points2

        Note: if there is a different amount of points in the two sets, only the first points are compared

        points1 (array): point set 1
        points2 (array): point set 2
        return (float): mean difference
        """
        n = min(points1.shape[0], points2.shape[0])
        return np.mean(np.linalg.norm(points1[:n,:]-points2[:n,:], axis=1))