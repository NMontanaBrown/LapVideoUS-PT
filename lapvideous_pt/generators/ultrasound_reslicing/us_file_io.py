# coding=utf-8

"""
Class to pre-process data for simulation from slicesampler
Warning! This code requires slicesampler, which requires
specific CUDA requirements.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import grid_sample as sample
import slicesampler.pycuda_simulation.segmented_volume as svol
import lapvideous_pt.generators.ultrasound_reslicing.us_generator as lvusg

class USTensorSlice():
    """
    Class to pre-process and load .npy volumes of
    features for tensor re-slicing.
    """
    def __init__(self,
                 mesh_dir,
                 config_dir,
                 voxel_size,
                 downsampling,
                 name_sim_tensor,
                 keys=None,
                 vol_type="multiclass"
                 ):
        """
        :param mesh_dir: str, directory with vtk models used in slicing.
        :param config_dir: str, json file with reslicing parameters and
        model names to be used.
        :param voxel_size: float, isotropic voxel size considered to
        generate the binary volumes for each vtk model.
        :param downsampling: int, downsampling factor on image dimensions.
        :param name_sim_tensor: str, name of file to save or load simulated tensor
        :param keys: List[str], list of keys to simulate from LUS json model.
        """
        self.mesh_dir = mesh_dir
        self.config_dir = config_dir
        self.voxel_size = voxel_size
        self.downsampling=downsampling
        self.name_sim_tensor = name_sim_tensor
        self.keys=keys
        self.vol_type=vol_type

        # Check if pre-existing simulated tensor exists
        if os.path.exists(os.path.join(mesh_dir, name_sim_tensor)):
            self.sim_tensor = np.load(os.path.join(mesh_dir, name_sim_tensor))
            self.origin = np.load(os.path.join(mesh_dir, name_sim_tensor.replace(".npy", "_origin.npy")))
            self.image_dim = np.load(os.path.join(mesh_dir, name_sim_tensor.replace(".npy", "_imdim.npy")))
            self.pixel_size = np.load(os.path.join(mesh_dir, name_sim_tensor.replace(".npy", "_pixdim.npy")))
        else:
            # Does not exist, convert slicesampler objects into npy tensor.x
            self.convert_to_tensor(mesh_dir, config_dir, voxel_size, downsampling, self.keys, self.vol_type)

    def convert_to_tensor(self,
                          mesh_dir,
                          config_dir,
                          voxel_size,
                          downsampling,
                          keys=None,
                          vol_type="multiclass"):
        """
        Use slicesampler to create a npy tensor.
        :param mesh_dir: str, directory with vtk models used in slicing.
        :param config_dir: str, json file with reslicing parameters and
        model names to be used.
        :param voxel_size: float, isotropic voxel size considered to
        generate the binary volumes for each vtk model.
        :param downsampling: int, downsampling factor on image dimensions.
        :param name_sim_tensor: str, name of file to save or load simulated tensor
        :param keys: List[str], list of keys to simulate from LUS json model.
        :param vol_type: str, type of tensor to generate. If "multiclass", the tensor
                         is generated with n_channels == len(keys), with each key
                         corresponding to the same position channel. Else, all the
                         features are added as binary voxels to the tensor.
        """
        segmented_volume = svol.SegmentedVolume(mesh_dir,
                                                config_dir,
                                                voxel_size=voxel_size,
                                                downsampling=downsampling,
                                                image_num=1)
        dict_info = segmented_volume.binary_volumes
        image_vars = segmented_volume.image_variables
        tensor, origin, _ = lvusg.generate_us_simulation_tensor(dict_info, voxel_size, keys, vol_type)
        self.sim_tensor = tensor
        self.origin = origin
        self.image_dim = image_vars[2]
        self.pixel_size = image_vars[3]

        # Save
        np.save(os.path.join(mesh_dir, self.name_sim_tensor), tensor)
        np.save(os.path.join(mesh_dir, self.name_sim_tensor.replace(".npy", "_origin.npy")), origin)
        np.save(os.path.join(mesh_dir, self.name_sim_tensor.replace(".npy", "_imdim.npy")), self.image_dim)
        np.save(os.path.join(mesh_dir, self.name_sim_tensor.replace(".npy", "_pixdim.npy")), self.pixel_size)

    def slice_and_compare(self, pose):
        """
        To check appropriate render, check against slicesampler.
        :param pose: str, path to GT slicesampler p2l pose.
        :return: void
        """
        seg_volume = svol.SegmentedVolume(self.mesh_dir,
                                          self.config_dir,
                                          voxel_size=self.voxel_size,
                                          downsampling=self.downsampling,
                                          image_num=1)
        _, binary_map_nc, _ =\
        seg_volume.simulate_image(poses=pose, image_num=1, out_points=True)

        out_im = lvusg.slice_volume(self.image_dim, self.pixel_size, pose, self.voxel_size, self.origin, self.sim_tensor, 1, "cpu")

        _, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(np.transpose(out_im.numpy()[0, :, 0, :, :], [1, 2, 0])) # B, W, H, D, Ch, (B, D = 1)
        axs[0].set_title("PyTorch Rendering")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].imshow(binary_map_nc[:, :, 0])
        axs[1].set_title("CUDA Rendering")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        plt.show()
