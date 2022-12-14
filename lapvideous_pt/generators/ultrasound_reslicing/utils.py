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
import sksurgerycore.algorithms.procrustes as p
from sksurgeryvtk.utils.matrix_utils import create_matrix_from_list

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
    :param normal: np.Array, (N, 3), defining normal.
    :param point: np.Array, (N, 3), defining coordinate.
    :return matrix: np.array (N, 4, 4), homogenous transform
                    in slicesampler format.

    References:
    [1]: J. Ramalhinho et al., "Registration of untracked 2D
         laparoscopic ultrasound to CT images of the liver
         using multi-labelled content-based image retrieval",
         DOI: 10.1109/TMI.2020.3045348
    """
    z_im = -np.tile(np.array([1, 0, 0]), (normal.shape[0], 1))
    y_im = -normal
    x_im = np.cross(y_im, z_im)
    matrix = np.tile(np.eye(4), (normal.shape[0], 1, 1))
    matrix[:, 0:3,0] = x_im
    matrix[:, 0:3, 1] = y_im
    matrix[:, 0:3, 2] = z_im
    matrix[:, 0:3, 3] = point
    return matrix

def find_closest_poses(pose, points):
    """
    For a given surface defined by points,
    and a given pose, find closest matches
    on points to pose.
    :param pose: np.array (4, 4), homogenous transform
                    in slicesampler format of LUS pose.
    :param point: np.Array, (N, 3), defining coordinates
                  of surface
    :return: ordered_points, np.Array, (N), indices of
             closest matches in points, ordered.
    """

    distance = np.linalg.norm(points - pose[0:3, 3], ord=2, axis=1)
    indices = np.argsort(distance)
    return indices, distance

def generate_close_poses(pose, points, normals, num_poses=10):
    """
    For a given surface defined by points and normals,
    and a given pose, automatically generate num_poses
    top closest matches slicesampler poses located
    at points coordinates.

    :param pose: np.array (4, 4), homogenous transform
                    in slicesampler format of LUS pose.
    :param points: np.Array, (N, 3), defining coordinates
                  of surface
    :param nomarls: np.Array, (N, 3), defining normals at
                    points.
    :param num_poses: int, number of closest matches to
                      define poses for
    :return np.array (num_poses, 4, 4), homogenous transform
                    in slicesampler format of new LUS poses
    """
    indices, distance = find_closest_poses(pose, points)
    new_poses_points = points[indices[:num_poses], :]
    new_poses_normals = normals[indices[:num_poses], :]
    new_poses = pose_constructor(new_poses_normals, new_poses_points)
    return new_poses, distance

def get_point_range_for_matrices(pose, sim_params):
    """
    For a given initial simulation pose, and a set
    of training simulation parameters, generate the
    coordinates that bound the object movement
    in it's frame of reference.
    :param pose: np.array, (4, 4) homogenous transform
                 defining original simulation pose.
    :param sim_params: list, (3) or (6), defining translation
                       simulation params +-tx, +-ty, +-tz in that
                       order. A (3) list will use each parameter
                       as the std on a zero-mean range.
    :return: np.array, [8, 3], coordinates defining
             bounding box of the movement for that pose. 
    """
    # List of l2c points
    if len(sim_params)==3:
        xx, yy, zz = np.meshgrid([-sim_params[0], sim_params[0]],
                                [-sim_params[1], sim_params[1]],
                                [-sim_params[2], sim_params[2]])
    else: # len 6
        xx, yy, zz = np.meshgrid([sim_params[0], sim_params[1]],
                                [sim_params[2], sim_params[3]],
                                [sim_params[4], sim_params[5]])
    xx_all = np.ravel(xx)
    yy_all = np.ravel(yy)
    zz_all = np.ravel(zz)
    points = []

    for i in range(xx_all.shape[0]):
        mat_t = create_matrix_from_list([0,0,0, xx_all[i], yy_all[i], zz_all[i]],
                                        is_in_radians=False)
        transformed_array = mat_t @ pose
        points.append(transformed_array[:3, 3])
    return np.array(points)
