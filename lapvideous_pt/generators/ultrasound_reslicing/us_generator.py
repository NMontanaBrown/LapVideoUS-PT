# coding=utf-8

"""
Pytorch Differentiable US rendering.
"""

import os
import random
from sys import path_importer_cache
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import grid_sample as sample

def generate_us_simulation_tensor(dict_arrays_slicesampler, voxel_size, keys=None, vol_type="multiclass"):
    """
    Generates a np.array of combined tensors from a slicesampler dictionary.
    We find the smallest origin and largest common volume shape, and pad the
    smaller tensors to the right shape and origin.

    :param dict_arrays_slicesampler: dict, of origins and npy volumes.
    :param voxel_size: float, size of voxels.
    :param keys: list, which models to concatenate into a tensor.
    :return: np.array
    """
    keys_dict = dict_arrays_slicesampler.keys()
    if keys:
        keys = [key for key in keys_dict if key in keys]
        if not keys:
            # Empty intersection, raise error
            raise ValueError("Passed keys does not match simulated tensors.")
    else:
        keys = keys_dict
    origin_list = []
    volume_shape_list = []
    volumes = []
    for key in keys:
        origin_list.append(dict_arrays_slicesampler[key][1][0,:])
        volumes.append(dict_arrays_slicesampler[key][0])
        volume_shape_list.append(dict_arrays_slicesampler[key][0].shape)
    # Turn into np arrays#
    volume_shapes = np.array(volume_shape_list)
    origins = np.array(origin_list)
    # Find smallest origin for each dimension
    min_origin = np.min(origins, axis=0)
    # Find largest shape to pad to
    min_vol_shapes = np.max(volume_shapes, axis=0)
    diff_origin = origins - min_origin
    shape_diff = min_vol_shapes - volume_shapes
    # Calculate how much we pad to the left and right of the volume in each dimension
    # Generally speaking this works, but we need to be careful in the case that a volume
    # is the max shape, and whose origin is not aligned with the min origin.
    # In this case we would expect the shape_diff == 0, and the diff_origin =/ 0.
    # In this special case, we would roll the dimensions without padding.
    # Aldso, 
    left_pad = np.array(diff_origin / voxel_size, dtype=np.int) # always +ve
    right_pad = np.array(shape_diff - left_pad, dtype=np.int) # can be -ve
    diff_pad = left_pad + right_pad
    roll_indices = np.where(right_pad<0, right_pad, 0)
    right_pad_neg_indices = np.argwhere(right_pad<0)
    # If negative indices, replace them with 0s
    right_pad[right_pad_neg_indices[:, 0], right_pad_neg_indices[:, 1]] = 0
    # For left pad, if sum of left, right_pad does not equal zero at neg indices,
    # We still want to left pad the volume and change the roll index.
    diff_pad_items = diff_pad[right_pad_neg_indices[:, 0], right_pad_neg_indices[:, 1]]
    indices_left_pad_pos = np.argwhere(diff_pad_items>0)
    positive_diff_pad_items = right_pad_neg_indices[indices_left_pad_pos[:, 0]]
    # replace left pad with positive diff pad items
    left_pad[positive_diff_pad_items[:, 0], positive_diff_pad_items[:, 1]] = diff_pad_items[indices_left_pad_pos[:, 0]]
    indices_left_pad_neg = np.argwhere(diff_pad_items<=0)
    negative_diff_pad_items = right_pad_neg_indices[indices_left_pad_neg[:, 0]]
    left_pad[negative_diff_pad_items[:, 0], negative_diff_pad_items[:, 1]] = 0
    new_volumes = []
    for index, volume in enumerate(volumes):
        # Pad and expand dims along channel
        padded_volume = np.pad(volume,
                               ((left_pad[index, 0], right_pad[index, 0]),
                                (left_pad[index, 1], right_pad[index, 1]),
                                (left_pad[index, 2], right_pad[index, 2])))
        final_volume = np.roll(padded_volume, (roll_indices[index, :]))
        new_volumes.append(np.expand_dims(final_volume, axis=-1))
        print(final_volume.shape)
    # Final tensor is the full volume
    if vol_type == "multiclass":
        volume_full = np.concatenate(new_volumes, axis=-1)
    else:
        volume_full = np.expand_dims(np.zeros(min_vol_shapes), -1)
        for item in new_volumes:
            volume_full += item
        volume_full = np.where(volume_full, 1, 0)
    max_origin = min_origin + (voxel_size*min_vol_shapes)

    return volume_full, min_origin, max_origin

def generate_cartesian_grid(im_x_size,
                            im_y_size,
                            im_x_res,
                            im_y_res,
                            batch,
                            device):
    """
    Generates a grid of cartesian coordinates.
    :param im_x_size: int, size of x grid
    :param im_y_size: int, size of y grid
    :param im_x_res: float, pixel x resolution
    :param im_y_res: float, pixel y resolution
    :param batch: int, batch size for generation.
    :return: [coordinates, torch.Tensor [B, Y, X, 1, 4]
              shape_planes, list [B, Y, X, 1, 4]
              shape_coords, list [B, Y, X, 1, 3]]
    """
    # Define cartesian grids
    x_grid = im_x_res*torch.range(start=-im_x_size/2, end=im_x_size/2-1, step=1, dtype=torch.float32, device=device)
    y_grid = im_y_res*torch.range(start=0, end=im_y_size-1, step=1, dtype=torch.float32, device=device)
    x_values, y_values = torch.meshgrid(y_grid, x_grid)
    zeros = torch.zeros((im_y_size, im_x_size, 1), dtype=torch.float32, device=device)
    ones = torch.ones((im_y_size, im_x_size, 1), dtype=torch.float32, device=device)
    # Convert into homogenous, coordinate tensor grid - xyz-1
    coordinates = torch.cat([torch.transpose(torch.transpose(y_values.expand(1, -1, -1), 0, 2), 0, 1),
                             torch.transpose(torch.transpose(x_values.expand(1, -1, -1), 0, 2), 0, 1),
                             zeros, ones], axis=-1) # [N, M, 4]
    return coordinates, [batch, im_y_size, im_x_size, 1, 4], [batch, im_y_size*im_x_size, 1, 3]


def planes_to_coordinates(coordinates, shape_planes, matrices):
    """
    Project planes into cartesian space based on poses
    :param coordinates: torch.Tensor
    :param shape_planes: list
    :param matrices: torch.Tensor
    :return: [coordinates [B], coordinates_planes]
    """
    # Reshape into (1, C, 4) matrix
    coords_vector = torch.reshape(coordinates, (-1, 4)).expand(1, -1, -1) # [1, C, 4]

    batch_coords = coords_vector.repeat(shape_planes[0], 1, 1) # [B, C, 4]
    # Matmul by matrices - [B, 4, 4], results in [B, C, 4] for each plane
    coordinates = torch.matmul(matrices,
                               torch.transpose(batch_coords, 1, 2))
    coordinates = torch.transpose(coordinates, 2, 1) # [B, C, 4]
    # Return coordinate planes - reshape to [B, N, M, 4]
    coordinates_planes = torch.reshape(coordinates, shape_planes) # [B, N, M, 1, 4]

    return coordinates, coordinates_planes


def generate_volume_coordinates(voxel_res,
                                origin_volume,
                                coordinates_planes,
                                shape_planes):
    """
    Generate the volume voxel indices corresponding to
    calculated plane coordinates from a pose.

    :param voxel_res: torch.Tensor, [3]
    :param origin_volume: torch.Tensor, [3]
    :param coordinates_planes: torch.Tensor, [B, N, M, 1, 4]
    :param shape_planes: shape of planes, [B, N, M, 1, 4]
    :return: voxel_locs, torch.Tensor, [B, N, M, 1, 3]
    """
    # getting rid of the homogenous part
    xyz_planes, _ = torch.split(coordinates_planes, [3, 1], -1) # [B, N, M, 1, 3]

    # Create [B, N, M, 1, 3] size voxel_dims, voxel_res, origin_volume
    voxel_res_ones = voxel_res.expand(1, 1, 1, 1, -1)
    batch_voxel_res = voxel_res_ones.repeat(shape_planes[0], shape_planes[1], shape_planes[2], 1, 1)
    origin_volume_ones = origin_volume.expand(1, 1, 1, 1, -1)
    batch_origin_volume = origin_volume_ones.repeat(shape_planes[0], shape_planes[1], shape_planes[2], 1, 1)

    # Calculate voxel indices of plane coordinates
    # Origin volume is coordinate of [0, 0, 0] voxel of volume
    # Therefore, voxel index is = (coordinate - origin) / resolution
    voxel_locs = torch.divide(torch.subtract(xyz_planes, batch_origin_volume),
                           batch_voxel_res)

    return voxel_locs # [B, M, N, 1, 3]

def normalise_voxel_locs(voxel_locs, shape_planes, grid_size, device):
    """
    Convert voxel_locs to voxel_locs in normal space [-1, 1] in
    x,y,z for use with pytorch.
    :param voxel_locs: np.array
    :param shape_planes:
    :param min_origin: np.array:
    :param max_origin: np.array
    """
    grid_size_repeat = torch.as_tensor(grid_size, dtype=torch.float32, device=device).repeat(
                                       shape_planes[0], shape_planes[1], shape_planes[2], 1, 1)
    grid_size_half = grid_size_repeat /2
    voxel_locs_norm = torch.divide(torch.subtract(voxel_locs, grid_size_half), grid_size_half) # [3]

    return voxel_locs_norm # [B, M, N, 1, 3]

def slice_volume(image_dim,
                 pixel_size,
                 pose,
                 voxel_size,
                 origin,
                 volume,
                 device):
    """
    Function to reslice a volume for a given pose in the
    volume.
    :param image_dim: list, (2,) image dimensions.
    :param pixel_size: list, (2,), pixel size.
    :param pose: torch.Tensor, (N, 4, 4), homogenouse pose in
                 slicesampler reference frame.
    :param voxel_size: like torch.Tensor, (1)
    :param origin: like torch.Tensor, (3,)
    :param volume: torch.Tensor, (N, W, H, D, Ch), tensor of 3D
                   volumetric features.
    :return: planes, torch.Tensor, (N, A, B) of resliced planes
             in tensor.
    """
    coordinates_orig, shape_planes, _ = generate_cartesian_grid(im_x_size=image_dim[0],
                                                                im_y_size=image_dim[1],
                                                                im_x_res=pixel_size[0],
                                                                im_y_res=pixel_size[1],
                                                                batch=1,
                                                                device=device)

    _, coordinates_planes = planes_to_coordinates(torch.as_tensor(coordinates_orig, dtype=torch.float32, device=device),
                                                  shape_planes,
                                                  matrices=torch.as_tensor(pose, dtype=torch.float32, device=device))

    # Voxel locs [B, M, N, 1, 3]
    voxel_locs = generate_volume_coordinates(voxel_res=torch.as_tensor([voxel_size, voxel_size, voxel_size], dtype=torch.float32, device=device),
                                             origin_volume=torch.as_tensor([origin[0], origin[1], origin[2]],dtype=torch.float32, device=device),
                                             coordinates_planes=coordinates_planes,
                                             shape_planes=shape_planes)

    voxel_locs_norm = normalise_voxel_locs(voxel_locs, shape_planes, volume.shape[0:3], device)

    voxel_locs_norm = torch.transpose(torch.transpose(voxel_locs_norm, 1, 3), 2, 3)
    # Create volume tensor
    volume = torch.transpose(torch.transpose(torch.transpose(torch.transpose(torch.as_tensor(volume, dtype=torch.float32, device=device).expand(1, -1, -1, -1, -1), 4, 1), 4, 2), 2, 3), 4, 3)

    out_im = sample(volume, voxel_locs_norm)
    return out_im
