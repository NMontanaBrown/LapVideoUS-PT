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
from pytorch3d.transforms import Transform3d

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
    outfile = os.path.splitext(file)[0]+ ".stl"
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file)
    reader.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetFileName(outfile)
    writer.Write()

def opencv_to_opengl(matrix_R, matrix_T, device):
    """
    Converts [N, 3, 3], [B, 3] left-handed (A = Bx)
    matrix in open-cv to [B, 4, 4]
    right-handed matrix. (A = xB)
    Useful if we have reference poses acquired in opencv
    that we need in Pytorch convention for rendering.
    Warning! The matrix must be a w2c matrix.
    :param matrix_R: torch.Tensor, [B, 3, 3] Rotation matrix.
    :param matrix_T: torch.Tensor, [B, 3], T vector
    :return: [left_handed_M, R_pytorch3d, T_pytorch3d]
    """
    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = matrix_R.clone().permute(0, 2, 1) # Transpose
    T_pytorch3d= matrix_T.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1
    eye = torch.eye(4, dtype=torch.float32, device=device) # Use last column
    _, column = torch.split(eye, [3, 1])
    column = torch.transpose(column.expand(1, -1, -1).repeat(R_pytorch3d.shape[0], 1, 1), 2, 1)
    left_handed_M = torch.cat((R_pytorch3d, torch.transpose(T_pytorch3d.expand(1, -1, -1,), 1, 0)), 1)
    left_handed_M = torch.cat((left_handed_M, column), 2)
    return left_handed_M, R_pytorch3d, T_pytorch3d

def opengl_to_opencv(matrix_R, matrix_T, device):
    """
    Converts [N, 3, 3], [N, 3] right-handed (A = xB)
    matrix in openGL/PyTorch to openCV
    left-handed matrix. (A = Bx)
    Warning! The matrix must be a w2c matrix.
    :param matrix_R: torch.Tensor, [B, 3, 3] Rotation matrix.
    :param matrix_T: torch.Tensor, [B, 3], T vector
    :return: [right_handed_M, R_openCV, T_openCV]
    """
    R_openCV = matrix_R.clone()
    T_openCV = matrix_T.clone()
    R_openCV[:, :, :2] *= -1
    T_openCV[:, :2] *= -1
    R_openCV = R_openCV.permute(0, 2, 1) # Transpose

    eye = torch.eye(4, dtype=torch.float32, device=device) # Use last column
    _, column = torch.split(eye, [3, 1])
    right_handed_M = torch.cat((R_openCV, torch.transpose(torch.transpose(T_openCV.expand(1, -1, -1), 2, 0), 1, 0)), 2)
    right_handed_M = torch.cat((right_handed_M, column.expand(1, -1, -1).repeat(R_openCV.shape[0], 1, 1)), 1)
    return right_handed_M, R_openCV, T_openCV

def opengl_to_opencv_p2l(matrix_R, matrix_T, device):
    """
    Specifically for converting OpenGL for liver
    frame of reference to openCV.
    :param matrix_R: torch.Tensor [B, 3, 3]
    :param matrix_T: torch.Tensor [B, 3,]
    """
    R_openCV = matrix_R.clone()
    T_openCV = matrix_T.clone()
    R_openCV = R_openCV.permute(0, 2, 1) # Transpose

    eye = torch.eye(4, dtype=torch.float32, device=device) # Use last column
    _, column = torch.split(eye, [3, 1])
    right_handed_M = torch.cat((R_openCV, torch.transpose(torch.transpose(T_openCV.expand(1, -1, -1), 2, 0), 1, 0)), 2)
    right_handed_M = torch.cat((right_handed_M, column.expand(1, -1, -1).repeat(R_openCV.shape[0], 1, 1)), 1)
    return right_handed_M, R_openCV, T_openCV

def opencv_to_opengl_p2l(matrix_R, matrix_T, device):
    """
    Specifically for converting OpenCV preds
    for liver frame of reference to OpenGL.
    :param matrix_R: torch.Tensor [B, 3, 3]
    :param matrix_T: torch.Tensor [B, 3,]
    """
    R_openGL = matrix_R.clone()
    T_openGL = torch.transpose(matrix_T.clone().expand(1, -1, -1), 1, 0) # (N, 1, 3)
    R_openGL = R_openGL.permute(0, 2, 1) # Transpose

    eye = torch.eye(4, dtype=torch.float32, device=device) # Use last column
    _, column = torch.split(eye, [3, 1])
    column = torch.transpose(column.expand(1, -1, -1).repeat(R_openGL.shape[0], 1, 1), 2, 1)
    left_handed_M = torch.cat((R_openGL, T_openGL), 1) # Cat along rows
    left_handed_M = torch.cat((left_handed_M, column), 2) # Cat along columns
    return left_handed_M, R_openGL, T_openGL

def split_opengl_hom_matrix(matrix):
    """
    OpenGL/Pytorch3D format = [R 0,
                               T 1]
    Splits the matrix into R and T formats.
    :param matrix: torch.Tensor, [N, 4, 4]
    :return: [R, T]
    """
    r, t = torch.split(torch.split(matrix, [3, 1], 2)[0], [3, 1], 1)
    t =  torch.squeeze(t, 1) # N, 3
    return r, t

def split_opencv_hom_matrix(matrix):
    """
    OpenCV format = [R T,
                     0 1]
    Splits the matrix into R and T formats.
    :param matrix: torch.Tensor, [N, 4, 4]
    :return: [R, T]
    """
    r, t = torch.split(torch.split(matrix, [3, 1], 1)[0], [3, 1], 2)
    t =  torch.squeeze(t, 2) # N, 3
    return r, t

def p2l_2_slicesampler(pose):
    """
    Function to convert p2l oepnCV output into slicesampler frame of reference.

    :param pose: torch.Tensor, (N,4,4) homogenous transformation
                 representing p2l output in openCV.
    :return: new_pose, torch.Tensor, (N,4,4) homogenous transformation
             of US plane characterization for slicesampler databases.
    """
    pose_clone = pose.clone()
    x, y, z, t = torch.split(pose_clone, [1, 1, 1, 1], 2)
    x_new = torch.neg(y)
    x_new_3, _ = torch.split(x_new, [3, 1], 1)
    z_3, z_0 = torch.split(z, [3, 1], 1)
    z_new = torch.cross(torch.transpose(x_new_3, 2, 1), torch.transpose(z_3, 2, 1))
    z_new = torch.cat([torch.transpose(z_new, 2, 1), z_0], dim=1)
    pose_slicesampler = torch.cat([x_new, z, z_new, t], dim=2)
    return pose_slicesampler

def slicesampler_2_p2l(pose):
    """
    Function to convert slicesampler pose into p2l frame of reference.
    Effectively revert p2l_2_slicesampler.

    :param pose: torch.Tensor, (N,4,4) homogenous transformation
             of US plane characterization for slicesampler databases.
    :return: torch.Tensor, (N,4,4) homogenous transformation
                 representing p2l output in openCV.
    """

    # Generate new frame of reference in p2l.
    # -x_slicesampler <- y_p2l, y_slicesampler <- z_p2l,
    # z_slicesampler = the cross of x_slicesampler and y_slicesampler
    x, y, z, t = torch.split(pose, [1, 1, 1, 1], 2)
    _, x_0 = torch.split(x, [3, 1], 1)
    y_old, y_0 = torch.split(torch.neg(x), [3, 1], 1)
    z_old, z_0 = torch.split(y, [3, 1], 1)
    x_old  = torch.cross(torch.transpose(y_old, 2, 1), torch.transpose(z_old, 2, 1))
    x_old = torch.cat([torch.transpose(x_old, 2, 1), x_0], dim=1)
    y_old_final = torch.cat([y_old, y_0], dim=1)
    z_old_final = torch.cat([z_old, z_0], dim=1)
    pose_slicesampler = torch.cat([x_old, y_old_final, z_old_final, t], dim=2)
    return pose_slicesampler

def global_to_local_space(pose_t, bounds):
    """
    Transforms pose in world space to a normalised
    world space.
    :param pose: torch.Tensor, (N, 3)
    :param bounds: torch.Tensor, (3)
    :return: torch.Tensor, (N, 3)
    """
    return torch.divide(pose_t, bounds)

def local_to_global_space(pose_t, bounds):
    """
    Transforms pose in a normalised world
    space to global world space.
    :param pose: torch.Tensor, (N, 3)
    :param bounds: torch.Tensor, (3)
    :return: torch.Tensor, (N, 3)
    """
    return torch.multiply(pose_t, bounds)

def generate_random_params_index(device, batch, index, start, stop):
    """
    Generate a torch.Tensor of values at index
    positions in list.
    :param device: 
    :param batch: int, N
    :param index: List[int], which indices to change.
    :param start: List[float]
    :param stop: List[float]
    :return: torch.Tensor, (N, 6)
    """
    # Generate (batch, 1) zeros or random numbers.
    list_tensors = []
    for i in range(6):
        if i in index:
            list_tensors.append((start[index.index(i)]-stop[index.index(i)])*torch.randn((batch, 1), device=device) + start[index.index(i)])
        else:
            list_tensors.append(torch.zeros((batch, 1), device=device))
    return torch.cat(list_tensors, dim=1)

def generate_ordered_params_index(device, batch, index, start, stop):
    """
    Generate a torch.Tensor of values at index
    positions with start, stop of len batch.
    :param device: 
    :param batch: int, N
    :param index: List[int], which indices to change
    :param start: List[float], start point
    :param end: List[float], end point of range
    :return: torch.Tensor, (N, 6)
    """
    # Generate (batch, 1) zeros or ordered list.
    list_tensors = []
    for i in range(6):
        if i in index:
            list_tensors.append(torch.transpose(torch.linspace(start=start[i],
                                                end=stop[i],
                                                steps=batch,
                                                device=device).expand(1, -1),
                                                1, 0))
        else:
            list_tensors.append(torch.zeros((batch, 1), device=device))
    return torch.cat(list_tensors, dim=1)

def generate_transforms(device, batch, tensor_params=None):
    """
    Generate a set of (N, 4, 4) torch.Tensors.
    :param device: torch.device
    :param batch: int, N
    :param tensor_params: torch.Tensor (N, 6)
    :return: Transform3d
    """
    if tensor_params is None:
        # Generate random values between -10 and 10
        tensor_params = (20)*torch.rand((batch, 6), device=device) + -10

    rot_x_vals, rot_y_vals, rot_z_vals, t_xyz =\
        torch.split(tensor_params, [1,1,1,3], 1)

    # Compose matrices
    transform_t = Transform3d(device=device).translate(t_xyz)
    transform_rot_y = Transform3d(device=device).rotate_axis_angle(
                                                    torch.squeeze(rot_y_vals), "Y")
    transform_rot_z = Transform3d(device=device).rotate_axis_angle(
                                                    torch.squeeze(rot_z_vals), "Z")
    transform_rot_x = Transform3d(device=device).rotate_axis_angle(
                                                    torch.squeeze(rot_x_vals), "X")
    rot_transforms = Transform3d(device=device).compose(transform_rot_y, transform_rot_x, transform_rot_z)
    return Transform3d(matrix=rot_transforms.get_matrix().double()), Transform3d(matrix=transform_t.get_matrix().double())

def perturb_orig_matrices(transform_l2c,
                          transform_p2c,
                          perturbations_c2l_r,
                          perturbations_c2l_t,
                          perturbations_p2l_r,
                          perturbations_p2l_t):
    """
    :param transform_l2c: Transform3d object
    :param transform_p2c: Transform3d object
    :param perturbations_c2l: Transform3d object
    :param perturbations_p2l: Transform3d object
    """
    # Perturb l2c by applying perturbations in world space to c2l, and invert.
    transform_l2c_perturbed = transform_l2c.inverse().compose(perturbations_c2l_r,
                                                    perturbations_c2l_t,
                                                    ).inverse()
    # Perturb position of probe by modifying p2l GT
    p2l = transform_p2c.compose(transform_l2c.inverse())
    perturbed_p2l = p2l.compose(perturbations_p2l_r, perturbations_p2l_t)
    # p2c = (l2c) @ (p2l) [Vertices] in OpenCV
    transform_p2c_perturbed = perturbed_p2l.compose(transform_l2c_perturbed)
    return transform_l2c_perturbed, transform_p2c_perturbed
