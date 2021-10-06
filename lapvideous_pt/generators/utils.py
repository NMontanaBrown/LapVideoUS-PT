# coding=utf-8

"""
Module for generator utils
"""

import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

def p2l_2_slicesampler(pose):
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

def slicesampler_2_p2l(pose):
    """
    Function to convert slicesampler pose into p2l frame of reference.
    Effectively revert p2l_2_slicesampler.

    :param pose: np.array, (4,4) homogenous transformation representing
                 US plane orientation for slicesampler databases.
    :return: new_pose, np.array, (4,4) homogenous transformation
             representing probe transformation p2l for SmartLiver
             GUI.
    """
    new_pose = np.zeros((4,4))
    # Generate new frame of reference in p2l.
    # -x_slicesampler <- y_p2l, y_slicesampler <- z_p2l,
    # z_slicesampler = the cross of x_slicesampler and y_slicesampler
    z_old = pose[:3, 1]
    y_old = -pose[:3, 0]
    x_old = np.cross(y_old, z_old)
    new_pose[:3, 0] = x_old
    new_pose[:3, 1] = y_old
    new_pose[:3, 2] = z_old
    new_pose[:, 3] = pose[:, 3]
    return new_pose

def opencv_to_opengl_extrinsics(pose):
    """
    Function to convert opencv extrinsics into opengl and viceversa.
    :param pose: np.array, (4,4) homogenous transformation representing
                 extrinsic parameters in OpenCV frame of reference.
    :return: pose_gl
    """
    rot = R.from_euler("x", 180, degrees=True)
    rotation_m = np.eye(4)
    rotation_m[0:3, 0:3] = rot.as_matrix()
    pose_gl = rotation_m @ pose
    return pose_gl

def non_norm_2_normalised_space(pose, offset):
    """
    Function to convert pose in non-normalised liver space
    into mean-centered liver space.

    :param pose: np.array, (N,4,4) homogenous transform representing
                 pose from non-normalised liver.
    :param offset: np.array, (4,4), rot+translation to mean center
                   from non-norm liver.
    :return: norm_space_pose
    """
    norm_space_pose = copy.deepcopy(pose)
    norm_space_pose[..., :3, 3] += offset[0:3, 3]
    return norm_space_pose

def global_to_local_space(pose, mean, sigma):
    """
    Function to convert a pose in a global space to a local,
    unit cube. EG, a pose with identity rotation and T= [0,0,0]
    would have a T_local of [-0.5, 0, 0] if it were in a
    local unit cube whose center is at [30, 0, 0] with total length
    of side=120mm. The cube local center is the local space origin.

    :param pose: np.array, (4,4) homogenous transform representing
    :param mean: np.array, (3,1) of origin of
                        local space in global coordinates.
    :param sigma: float, half the cube length in global space in mm
    :return: unit_space_pose, np.array, (N,4,4) homogenous transform
             representing R+T relative to local, unit length cube bounding
             space.
    """
    unit_space_pose = copy.deepcopy(pose)
    unit_space_pose[..., :3, 3] -= mean[:, 0]
    unit_space_pose[..., :3, 3] = np.divide(unit_space_pose[..., :3, 3], sigma)
    return unit_space_pose

def local_to_global_space(unit_space_pose, mean, sigma):
    """
    Function to convert a pose from a local, normalise space
    into global space, as defined by the center of the unit length
    cube and the isotropic size (length) of the cube in mm.
    :param unit_space_pose: np.array, (N,4,4) homogenous transform
                             representing R+T relative to local,
                             unit cube bounding space.
    :param mean: np.array, (3,1), origin of
                        local space in global coordinates.
    :param sigma: float, half the cube length in global space in mm.
    :return: pose, np.array, (N,4,4) homogenous transform representing
             R+T in global space.
    """
    pose = copy.deepcopy(unit_space_pose)
    pose[..., :3, 3] *= sigma
    pose[..., :3, 3] += mean[:, 0]
    return pose

def slicesampler_gt_2_p2l(pose, offset):
    """
    Function to convert ground truth pose in non-normalised
    liver space into p2l representation.

    :param pose: np.array, (4,4) homogenous transform representing
                 pose from slicesampler in non normalised space.
    :param offset: np.array, (4,4), translation to mean center.
    :return: p2l_norm
    """
    p2l_non_norm = slicesampler_2_p2l(pose)
    p2l_norm = non_norm_2_normalised_space(p2l_non_norm,
                                           offset)
    return p2l_norm

def generate_split_grid_indices(training_splits:list,
                                grid_dims:tuple,
                                seed=None):
    """
    Generates lists of indices in grid. Splits them
    according to % of data required in train, val, test
    splits.

    :param training_splits: list, len(3), defining
                            train., val, test splits in %.
    :param grid_dims: tuple, len(6), of integers, defining number
                      of steps.
    :param seed: int, which random seed to use, (None).
    :return:
        - train: list of unique indices to use during training.
        - val: list of unique indices to use for validation.
        - test: list of unique indices to use for testing.
    """
    # Set seed
    np.random.seed(seed)
    # Get indices
    index_product = np.prod(list(grid_dims))
    array_all_indices = np.arange(0, index_product)
    # Shuffle
    np.random.shuffle(array_all_indices)
    test = np.random.choice(array_all_indices,
                            size=int(training_splits[-1]*index_product),
                            replace=False)
    train = set(array_all_indices.tolist()) - set(test.tolist())
    val = np.random.choice(np.asarray(list(train)),
                           size=int(training_splits[1]*index_product),
                           replace=False)
    train = train - set(val.tolist())

    return np.asarray(list(train)), val, test

def indices_to_pose_lists(indices:list,
                          grid_dims:tuple,
                          range_list:list,
                          step_list:list,):
    """
    Convert a list of indices in a given grid
    to a set of poses.
    :param index: List[int], number of index in grid
    :param grid_dims: tuple, size of grid in number of steps.
    :param range_list: list, starting point in space of grid (len griddims).
    :param step_list: list, step size for each dimension (len griddims).

    :return: poses, list of lists, len (indices), each list len (griddims),
             that can be used to simulate data.
    """
    # Get indices: returns grid_dims # arrays, each len indices (N).
    tuple_indices = np.unravel_index(indices, grid_dims)
    # Convert to N x Griddims
    poses = np.vstack(tuple_indices).T
    step_list_stack = np.tile(np.asarray(step_list), (len(indices), 1))
    range_list_stack = np.tile(np.asarray(range_list), (len(indices), 1))
    # Get final matrix
    final_pose_array = range_list_stack + np.multiply(poses, step_list_stack)
    # Unpack to lists
    return final_pose_array.tolist()
