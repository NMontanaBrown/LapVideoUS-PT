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
    # Turn into np arrays
    volume_shapes = np.array(volume_shape_list)
    origins = np.array(origin_list)
    # Find smallest origin for each dimension
    min_origin = np.min(origins, axis=0)
    # Find largest shape to pad to
    min_vol_shapes = np.max(volume_shapes, axis=0)
    diff_origin = origins - min_origin
    shape_diff = min_vol_shapes - volume_shapes
    # Calculate how much we pad to the left and right of the volume in each dimension
    left_pad = np.array(diff_origin / voxel_size, dtype=np.int)
    right_pad = np.array(shape_diff - left_pad, dtype=np.int)
    new_volumes = []
    for index, volume in enumerate(volumes):
        # Pad and expand dims along channel
        new_volumes.append(np.expand_dims(np.pad(volume,
                                                ((left_pad[index, 0], right_pad[index, 0]),
                                                (left_pad[index, 1], right_pad[index, 1]), 
                                                (left_pad[index, 2], right_pad[index, 2])) ),
                                          axis=-1))
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
                            batch,):
    """
    Generates a grid of cartesian coordinates.
    :param im_x_size: int, size of x grid
    :param im_y_size: int, size of y grid
    :param im_x_res: float, pixel x resolution
    :param im_y_res: float, pixel y resolution
    :param batch: int, batch size for generation.
    :return: [coordinates, tf.Tensor [B, Y, X, 1, 4]
              shape_planes, list [B, Y, X, 1, 4]
              shape_coords, list [B, Y, X, 1, 3]]
    """
    # Define cartesian grids
    x_grid = im_x_res*torch.range(start=-im_x_size/2, end=im_x_size/2-1, step=1, dtype=torch.float32)
    y_grid = im_y_res*torch.range(start=0, end=im_y_size-1, step=1, dtype=torch.float32)
    x_values, y_values = torch.meshgrid(y_grid, x_grid)
    zeros = torch.zeros((im_y_size, im_x_size, 1), dtype=torch.float32)
    ones = torch.ones((im_y_size, im_x_size, 1), dtype=torch.float32)
    # Convert into homogenous, coordinate tensor grid - xyz-1
    coordinates = torch.cat([torch.transpose(torch.transpose(y_values.expand(1, -1, -1), 0, 2), 0, 1),
                             torch.transpose(torch.transpose(x_values.expand(1, -1, -1), 0, 2), 0, 1),
                             zeros, ones], axis=-1) # [N, M, 4]
    return coordinates, [batch, im_y_size, im_x_size, 1, 4], [batch, im_y_size*im_x_size, 1, 3]


def planes_to_coordinates(coordinates, shape_planes, matrices):
    """
    Project planes into cartesian space based on poses
    :param coordinates: tf Tensor
    :param shape_planes: list
    :param matrices: tf Tensor
    :return: [coordinates [B], coordinates_planes]
    """
    # Reshape into (1, C, 4) matrix
    coords_vector = torch.reshape(coordinates, (-1, 4)).expand(1, -1, -1) # [1, C, 4]

    batch_coords = torch.tile(coords_vector, [shape_planes[0], 1, 1]) # [B, C, 4]
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

    :param voxel_res: tf.Tensor, [3]
    :param origin_volume: tf.Tensor, [3]
    :param coordinates_planes: tf.Tensor, [B, N, M, 1, 4]
    :param shape_planes: shape of planes, [B, N, M, 1, 4]
    :return: voxel_locs, tf.Tensor, [B, N, M, 1, 3]
    """
    # getting rid of the homogenous part
    xyz_planes, _ = torch.split(coordinates_planes, [3, 1], -1) # [B, N, M, 1, 3]

    # Create [B, N, M, 1, 3] size voxel_dims, voxel_res, origin_volume
    voxel_res_ones = voxel_res.expand(1, 1, 1, 1, -1)
    batch_voxel_res = torch.tile(voxel_res_ones, [shape_planes[0], shape_planes[1], shape_planes[2], 1, 1])
    origin_volume_ones = origin_volume.expand(1, 1, 1, 1, -1)
    batch_origin_volume = torch.tile(origin_volume_ones, [shape_planes[0], shape_planes[1], shape_planes[2], 1, 1])

    # Calculate voxel indices of plane coordinates
    # Origin volume is coordinate of [0, 0, 0] voxel of volume
    # Therefore, voxel index is = (coordinate - origin) / resolution
    voxel_locs = torch.divide(torch.subtract(xyz_planes, batch_origin_volume),
                           batch_voxel_res)

    return voxel_locs # [B, M, N, 1, 3]

def normalise_voxel_locs(voxel_locs, shape_planes, grid_size):
    """
    Convert voxel_locs to voxel_locs in normal space [-1, 1] in
    x,y,z for use with pytorch.
    :param voxel_locs: np.array
    :param shape_planes:
    :param min_origin: np.array:
    :param max_origin: np.array
    """
    grid_size_repeat = torch.tile(torch.as_tensor(grid_size, dtype=torch.float32),
                                  [shape_planes[0], shape_planes[1], shape_planes[2], 1, 1])
    grid_size_half = grid_size_repeat /2
    voxel_locs_norm = torch.divide(torch.subtract(voxel_locs, grid_size_half), grid_size_half) # [3]

    return voxel_locs_norm # [B, M, N, 1, 3]



### Test
key_word = "multi"
path_non_centered_vtk = os.path.abspath("/Users/nmont/PhD/ClinicalData/Data/DataSeg/VTK/2018-06-15_i4i_HLR_01/")
files = os.listdir(path_non_centered_vtk)
full_files = [os.path.join(path_non_centered_vtk, file) for file in files]

origin_file = [file for file in full_files if ("origin" in file) and (key_word in file)]
image_dim_file = [file for file in full_files if ("imdim" in file) and (key_word in file)]
volume_file = [file for file in full_files if ("channel_tensor.npy" in file) and (key_word in file)]
pixel_file = [file for file in full_files if ("pixdim" in file) and (key_word in file)]

# Known pose

path_GT_poses = "/Users/nmont/PhD/ClinicalData/Data/GT_Registrations/2018-06-15_i4i_HLR_01/2018.06.15_11-35-55-029_RightLobe"
path_non_centered_vtk = os.path.abspath("/Users/nmont/PhD/ClinicalData/Data/DataSeg/VTK/2018-06-15_i4i_HLR_01/")
folder_pose = '/poses/pose1/'
path_json =  os.path.abspath("/Users/nmont/PhD/ClinicalData/LapVideoUS/H01"+folder_pose)
model_lus_json = os.path.join(path_json, 'models_lus.json')
model_lus_non_centered = os.path.join(path_json, 'models_lus_non_centered.json')
pose_file = '/pose_651.txt'

pose_orig = np.loadtxt(path_GT_poses+pose_file, delimiter=',')
origin = np.load(origin_file[0])
volume = np.load(volume_file[0])
image_dim = np.load(image_dim_file[0])
pixel_size=np.load(pixel_file[0])

coordinates_orig, shape_planes, shape_coords = generate_cartesian_grid(im_x_size=image_dim[0],
                                                                        im_y_size=image_dim[1],
                                                                        im_x_res=pixel_size[0],
                                                                        im_y_res=pixel_size[1],
                                                                        batch=1,)

coordinates, coordinates_planes = planes_to_coordinates(torch.as_tensor(coordinates_orig, dtype=torch.float32),
                                                        shape_planes,
                                                        matrices=torch.as_tensor(pose_orig, dtype=torch.float32))

# Voxel locs [B, M, N, 1, 3]
voxel_locs = generate_volume_coordinates(voxel_res=torch.as_tensor([0.5, 0.5, 0.5], dtype=torch.float32),
                                        origin_volume=torch.as_tensor([origin[0], origin[1], origin[2]],dtype=torch.float32),
                                        coordinates_planes=coordinates_planes,
                                        shape_planes=shape_planes)

voxel_locs_norm = normalise_voxel_locs(voxel_locs, shape_planes, volume.shape[0:3])

voxel_locs_norm = torch.transpose(torch.transpose(voxel_locs_norm, 1, 3), 2, 3)
# Create volume tensor
volume = torch.transpose(torch.transpose(torch.transpose(torch.transpose(torch.as_tensor(volume, dtype=torch.float32).expand(1, -1, -1, -1, -1), 4, 1), 4, 2), 2, 3), 4, 3)

out_im = sample(volume, voxel_locs_norm)

fig, axs = plt.subplots(nrows=1, ncols=1)
axs.imshow(np.transpose(out_im.numpy()[0, :, 0, :, :], [1, 2, 0]))
plt.show()