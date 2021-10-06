# coding=utf-8

"""
Utils for differentiable video rendering, specifically useful for
file-io prior to rendering and training.
"""

import os
import vtk
import torch
import numpy as np
import vtk.numpy_interface.dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy

def get_points_from_vtk(file):
    """
    Read vtk file and return coordinates
    :param file: file path, .vtk extension
    :return: vertices. np.array, [N, 3]
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file)
    reader.Update()
    polydata = reader.GetOutput()

    vertices = vtk_to_numpy(dsa.WrapDataObject(polydata).Points)
    return vertices

def convert_points_to_stl(file):
    """
    Read .vtk file, and re-write as an .stl
    file.
    :param file: str, path to .vtk file to rewrite to .stl.
    :return: void
    """
    # Rename file with .stl ending.
    outfile = os.path.join(os.path.splitext(file)[0], ".stl")
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file)
    reader.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetFileName(outfile)
    writer.Write()

def opencv_to_opengl(matrix_R, matrix_T):
    """
    Converts [B, 4, 4] left-handed (A = Bx)
    matrix in open-cv to [B, 4, 4]
    right-handed matrix. (A = xB)
    Useful if we have reference poses acquired in opencv
    that we need in Pytorch convention for rendering.
    :param matrix_R: torch.Tensor, [B, 3, 3] Rotation matrix.
    :param matrix_T: torch.Tensor, [B, 3], T vecotr
    :return: [left_handed_M, R_pytorch3d, T_pytorch3d]
    """
    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = matrix_R.clone().permute(0, 2, 1) # Transpose
    T_pytorch3d= matrix_T.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    ones = torch.from_numpy(np.ones((1,4), np.float32)) # [1, 4]
    left_handed_M = torch.cat((R_pytorch3d, T_pytorch3d.expand(1, -1, -1,)), 1)
    left_handed_M = torch.cat((left_handed_M, torch.transpose(ones.expand(1, -1, -1), 2, 1)), 2)
    return left_handed_M, R_pytorch3d, T_pytorch3d
