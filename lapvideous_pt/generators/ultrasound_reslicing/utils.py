# coding=utf-8

"""
Module for utilities for ultrasound reslicing and generating
poses for simulation.
"""

import os
import numpy as np
import vtk
import vtk.numpy_interface.dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy


def get_model_vertices_faces_normals(path_vtk):
    """
    From path_vtk to .vtk object, read the file
    and extract coordinates of surface and
    normals to those points.
    :param path_vtk: str
    :return: vertices, np.Array, [N, 3]
             faces, np.Array, [M, 3]
             normals, np.Array, [N, 3]
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path_vtk)
    reader.Update()
    polydata = reader.GetOutput()
    # Extract faces and vertices
    vertices = vtk_to_numpy(dsa.WrapDataObject(polydata).Points)
    faces = vtk_to_numpy(dsa.WrapDataObject(polydata).Polygons)
    normals = vtk_to_numpy(polydata.GetPointData().GetNormals())
    return vertices, faces, normals

def pose_constructor(normal, point):
    """
    Given a coordinate point, and
    point normal, we define a pose on
    the conventions defined in [1] for simulation
    of US slices.
    :param normal: np.Array, (3), defining normal.
    :param point: np.Array, (3), defining coordinate.
    :return matrix: np.array (4, 4), homogenous transform
                    in slicesampler format.

    References:
    [1]: J. Ramalhinho et al., "Registration of untracked 2D
         laparoscopic ultrasound to CT images of the liver
         using multi-labelled content-based image retrieval",
         DOI: 10.1109/TMI.2020.3045348
    """
    z_im = -np.array([1, 0, 0])
    y_im = -normal
    x_im = np.cross(y_im, z_im)
    matrix = np.eye(4)
    matrix[0:3,0] = x_im
    matrix[0:3, 1] = y_im
    matrix[0:3, 2] = z_im
    matrix[0:3, 3] = point
    return matrix
