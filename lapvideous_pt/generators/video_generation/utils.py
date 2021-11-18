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

def constrain_quat_hemisphere(quaternion):
    """
    Rotation matrices have two solutions in
    quaternion notation q and -q. We can make the
    solution unique by constraining the quaternions
    to one of the hemispheres.
    :param quaternion: torch.Tensor, (N, 4)
    :return: quaternion_constrained, (N, 4)
    """
    # Constrain first element of quaternion to
    # always be positive.
    r, i, j, k = torch.split(quaternion, [1, 1, 1, 1], dim=1)
    sign_r = torch.sign(r)
    # Give sign of first element.
    # Multiply all i, j, k by sign to get quaternions in same
    # hemispheres.
    quaternion_constrained = torch.mul(quaternion, sign_r)
    return quaternion_constrained

def batch_p2l_2_slicesampler(poses):
    """
    Function to convert a batch of p2l poses into slicesampler representation,
    which requires that poses be stacked horizontally, as well as
    transformed into the slicesampler frame of reference.

    :param poses: np.array, (N, 4, 4) of homogenous transformations
                 representing p2l output of SmartLiver surface_probe_app.
    :return: new_poses, np.array, (4, 4*N)
    """
    new_poses = []
    for i in range(poses.shape[0]):
        new_poses.append(p2l_2_slicesampler_numpy(poses[i, :, :]))
    # convert from (N, 4, 4) to (4, N*4), by hstacking.
    stacked_poses = np.hstack(new_poses)
    return stacked_poses

def p2l_2_slicesampler_numpy(pose):
    """
    Function to convert p2l output into slicesampler frame of reference.

    :param pose: np.array, (4,4) homogenous transformation
                    representing p2l output of SmartLiver surface_probe_app.
    :return: new_pose, np.array, (4,4) homogenous transformation
                of US plane characterization for slicesampler databases.
    """
    new_pose = np.zeros((4,4))
    # Generate new frame of reference
    # x_slicesampler -> -y_p2l, y_slicesampler -> z_p2l,
    # z_slicesampler = the cross of x_slicesampler and y_slicesampler
    x_new = -pose[:3, 1]
    y_new = pose[:3, 2]
    z_new = np.cross(x_new, y_new)

    new_pose[:3, 0] = x_new
    new_pose[:3, 1] = y_new
    new_pose[:3, 2] = z_new
    new_pose[:, 3] = pose[:, 3] # translation is the same.
    return new_pose

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
    right_handed_M = cat_opencv_hom_matrix(R_openCV, T_openCV, device)

    return right_handed_M, R_openCV, T_openCV

def opencv_to_opengl_p2l(matrix_R, matrix_T, device):
    """
    Specifically for converting OpenCV preds
    for liver frame of reference to OpenGL.
    :param matrix_R: torch.Tensor [B, 3, 3]
    :param matrix_T: torch.Tensor [B, 3,]
    """
    R_openGL = matrix_R.clone()
    R_openGL = R_openGL.permute(0, 2, 1) # Transpose
    T_openGL = matrix_T.clone()
    left_handed_M = cat_opengl_hom_matrix(R_openGL, T_openGL, device)

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

def cat_opengl_hom_matrix(matrix_R, matrix_T, device):
    """
    OpenGL/Pytorch3D format = [R 0,
                               T 1]
    Concatenates R and T formats into hom matrix.
    :param r_matrix:
    :param t_matrix:
    :return: [N, 4, 4]
    """
    T_openGL = torch.transpose(matrix_T.clone().expand(1, -1, -1), 1, 0) # (N, 1, 3)
    R_openGL = matrix_R.clone()

    eye = torch.eye(4, dtype=torch.float32, device=device) # Use last column
    _, column = torch.split(eye, [3, 1])
    column = torch.transpose(column.expand(1, -1, -1).repeat(R_openGL.shape[0], 1, 1), 2, 1)
    left_handed_M = torch.cat((R_openGL, T_openGL), 1) # Cat along rows
    left_handed_M = torch.cat((left_handed_M, column), 2) # Cat along columns
    return left_handed_M

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

def cat_opencv_hom_matrix(matrix_R, matrix_T, device):
    """
    OpenCV format = [R T,
                     0 1]
    Concatenates R and T formats into hom matrix.
    :param r_matrix:
    :param t_matrix:
    :return: [N, 4, 4]
    """
    eye = torch.eye(4, dtype=torch.float32, device=device) # Use last column
    _, column = torch.split(eye, [3, 1])
    right_handed_M = torch.cat((matrix_R, torch.transpose(torch.transpose(matrix_T.expand(1, -1, -1), 2, 0), 1, 0)), 2)
    right_handed_M = torch.cat((right_handed_M, column.expand(1, -1, -1).repeat(matrix_R.shape[0], 1, 1)), 1)
    return right_handed_M

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
    return torch.div(pose_t, bounds)

def local_to_global_space(pose_t, bounds):
    """
    Transforms pose in a normalised world
    space to global world space.
    :param pose: torch.Tensor, (N, 3)
    :param bounds: torch.Tensor, (3)
    :return: torch.Tensor, (N, 3)
    """
    return torch.mul(pose_t, bounds)

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
            random_nums = torch.rand((batch, 1), device=device)
            list_tensors.append((stop[index.index(i)]-start[index.index(i)])*random_nums + start[index.index(i)])
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
            list_tensors.append(torch.transpose(torch.linspace(start=start[index.index(i)],
                                                end=stop[index.index(i)],
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
    transform_t = Transform3d(device=device, dtype=torch.float32).translate(t_xyz)
    transform_rot_y = Transform3d(device=device, dtype=torch.float32).rotate_axis_angle(
                                                    torch.squeeze(rot_y_vals), "Y")
    transform_rot_z = Transform3d(device=device, dtype=torch.float32).rotate_axis_angle(
                                                    torch.squeeze(rot_z_vals), "Z")
    transform_rot_x = Transform3d(device=device, dtype=torch.float32).rotate_axis_angle(
                                                    torch.squeeze(rot_x_vals), "X")
    rot_transforms = Transform3d(device=device, dtype=torch.float32).compose(transform_rot_y, transform_rot_x, transform_rot_z)
    return Transform3d(matrix=rot_transforms.get_matrix()), Transform3d(matrix=transform_t.get_matrix())

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

def perturb_orig_matrices_in_CV_space(transform_l2c,
                                      transform_p2c,
                                      perturbations_l2c_r,
                                      perturbations_l2c_t,
                                      perturbations_p2l_r,
                                      perturbations_p2l_t,
                                      device):
    """
    To use CV conventions, we convert l2c objects in GL->CV space,
    apply transformations in that space, and convert back to GL space.

    :param transform_l2c: Transform3d object
    :param transform_p2c: Transform3d object
    :param perturbations_l2c_r: Transform3d object
    :param perturbations_l2c_t: Transform3d object
    :param perturbations_p2l_r: Transform3d object
    :param perturbations_p2l_t: Transform3d object
    :return: [transform_l2c_perturbed, transform_p2c_perturbed]
    """
    transform_l2c_m = transform_l2c.get_matrix()
    transform_p2l_m = transform_p2c.compose(transform_l2c.inverse()).get_matrix()

    # Convert into CV space
    p2l_r_gl, p2l_t_gl = split_opengl_hom_matrix(transform_p2l_m)
    l2c_r_gl, l2c_t_gl = split_opengl_hom_matrix(transform_l2c_m)
    l2c_pert_r_r, l2c_pert_r_t = split_opengl_hom_matrix(perturbations_l2c_r.get_matrix())
    l2c_pert_t_r, l2c_pert_t_t = split_opengl_hom_matrix(perturbations_l2c_t.get_matrix())
    p2l_pert_r_r, p2l_pert_r_t = split_opengl_hom_matrix(perturbations_p2l_r.get_matrix())
    p2l_pert_t_r, p2l_pert_t_t = split_opengl_hom_matrix(perturbations_p2l_t.get_matrix())

    p2l_cv, _, _ = opengl_to_opencv_p2l(p2l_r_gl, p2l_t_gl, device)
    l2c_cv, _, _ = opengl_to_opencv(l2c_r_gl, l2c_t_gl, device)
    l2c_cv_r, _, _ = opengl_to_opencv_p2l(l2c_pert_r_r, l2c_pert_r_t, device)
    l2c_cv_t, _, _ = opengl_to_opencv_p2l(l2c_pert_t_r, l2c_pert_t_t, device)
    p2l_cv_r, _, _ = opengl_to_opencv_p2l(p2l_pert_r_r, p2l_pert_r_t, device)
    p2l_cv_t, _, _ = opengl_to_opencv_p2l(p2l_pert_t_r, p2l_pert_t_t, device)

    # Transform
    l2c_cv_pert = l2c_cv_t @ l2c_cv @ l2c_cv_r
    p2l_cv_pert = p2l_cv_t @ p2l_cv @ p2l_cv_r

    # Convert into GL space
    l2c_cv_r, l2c_cv_t = split_opencv_hom_matrix(l2c_cv_pert)
    p2l_cv_r, p2l_cv_t = split_opencv_hom_matrix(p2l_cv_pert)
    transform_l2c_perturbed, _, _ = opencv_to_opengl(l2c_cv_r, l2c_cv_t, device)
    transform_p2l_perturbed, _, _ = opencv_to_opengl_p2l(p2l_cv_r, p2l_cv_t, device)
    transform_l2c_perturbed = Transform3d(matrix=transform_l2c_perturbed)
    transform_p2c_perturbed = Transform3d(matrix=transform_p2l_perturbed).compose(transform_l2c_perturbed)
    return transform_l2c_perturbed, transform_p2c_perturbed
