# coding=utf-8

"""
Utils for 
"""

import os
import json
import numpy as np
import scipy.io as sio


def pose_2_quiver(pose):
    """
    Function to generate quiver arrows for MATLAB, using quiver3
    argument.

    :param pose: np.array, (4,4) homogenous transformation representing
                 pose.
    :return: np.array, (12,) given coordinates of quiver arrow for MATLAB vis.
    """
    # Save array
    quiver_save = np.zeros(12)
    # First three elements are the origin of the frame of reference
    # (ie T in pose)
    quiver_save[0:3] = pose[0:3, 3]
    # the vector describing 'x' is the first column of the
    # rotation matrix
    quiver_save[3:6] = pose[:3, 0]
    #  the vector describing 'y' is the second column of the
    # rotation matrix
    quiver_save[6:9] = pose[:3, 1]
    #  the vector describing 'z' is the third column of the
    # rotation matrix
    quiver_save[9:] = pose[:3, 2]
    # Save it.
    return quiver_save

def make_mat_file_mesh(vertices, path_save, file_name, var_name):
    """
    Function to generate vertex .mat file for visualisation of a certain plane
    in MATLAB
    :param vertices: np.array, vertices of surface.
    :param path_save: str, where to save file.
    :param file_name: str, name of file.
    :param var_name: str, name of variable for MATLAB
    """
    sio.savemat(os.path.join(path_save, file_name),
                {var_name: vertices})

def make_mat_files_plane(colored_map,
                         points,
                         path_save:str,
                         file_name:str,
                         var_name:str):
    """
    Function to generate .mat files for
    visualisation of a certain plane
    in MATLAB

    :param colored_map: np.array, output of slicesampler simulate image
                        func for given pose
    :param points: np.array, output of slicesampler simulate image func
                  for given pose
    :param path_save: str, path to save files
    :param file_name: str, .mat extension, name for file
    :param var_name: str, name of param for MATLAB
    """
    points_per_plane = int(points.shape[0]/colored_map.shape[3])
    plane_save = points[:points_per_plane:10, :]
    make_mat_file_mesh(plane_save, path_save, file_name, var_name)
