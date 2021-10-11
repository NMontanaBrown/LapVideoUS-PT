# coding=utf-8

"""
Create model to optimise the position of a camera wrt
to a GT pose.
"""

import os
from ntpath import join
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import Transform3d
# LapVideoUS-PT
import lapvideous_pt.generators.video_generation.utils as vru
import lapvideous_pt.generators.ultrasound_reslicing.us_generator as lvusg
from lapvideous_pt.generators.video_generation.video_generator import VideoLoader
import lapvideous_pt.generators.video_generation.mesh_utils as lvvmu

class LapVideoUS(nn.Module):
    def __init__(self,
                 mesh_dir,
                 config_dir,
                 path_us_tensors,
                 name_tensor,
                 liver2camera_reference,
                 probe2camera_reference,
                 intrinsics,
                 image_size,
                 output_size,
                 batch,
                 device):
        """
        Class that contains functions to synthetically
        render US and video differentiably
        using reference meshes and reference poses.
        :param mesh_dir:
        :param config_dir:
        :param path_us_tensors:
        :param name_tensor:
        :param liver2camera_reference:
        :param probe2camera_reference:
        :param intrinsics:
        :param image_size:
        :param output_size:
        :param batch:
        :param device:
        """
        super().__init__()
        # Setup CUDA device.
        if torch.cuda.is_available():
            device = torch.device(device)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        self.meshes_dict = mesh_dir
        self.device = device
        self.batch = batch
        self.pre_process_video_files(mesh_dir,
                                     config_dir,
                                     liver2camera_reference,
                                     probe2camera_reference,
                                     image_size,
                                     output_size,
                                     intrinsics,
                                     device)
        self.pre_process_US_files(path_us_tensors,
                                  name_tensor)

        # Would build NN layers here

    def pre_process_video_files(self,
                                mesh_dir,
                                config_dir,
                                liver2camera_reference,
                                probe2camera_reference,
                                image_size,
                                output_size,
                                intrinsics,
                                device):
        """
        Pre-process the video data and files

        """
        video_loader = VideoLoader(mesh_dir,
                                   config_dir,
                                   liver2camera_reference,
                                   probe2camera_reference,
                                   intrinsics,
                                   device)
        self.video_loader = video_loader
        self.video_loader.pre_process_reference_poses(self.video_loader.liver2camera_ref,
                                                      self.video_loader.probe2camera_ref)
        self.video_loader.load_meshes(mesh_dir, self.video_loader.config)
        np_intrinsics = np.loadtxt(intrinsics)
        self.video_loader.setup_renderer(np_intrinsics, image_size, output_size)
        self.bounds = torch.from_numpy(np.array([500, 500, 500])).to(self.device)

    def pre_process_US_files(self, path_us_tensors, name_tensor):
        """
        Pre-process data us data for rendering.
        We assume that the files all have the same root name_tensor,
        and that they live in path_us_tensors directory.
        :param path_us_tensors: str, path to folder containing the
                                pre-processed US data.
        :param name_tensor: str, name of simulation tensor to use.
        :return: void.
        """
        volume = torch.from_numpy(np.load(os.path.join(path_us_tensors, name_tensor))).to(device=self.device)
        origin = torch.from_numpy(np.load(os.path.join(path_us_tensors, name_tensor.replace(".npy", "_origin.npy")))).to(device=self.device)
        pix_dim = torch.from_numpy(np.load(os.path.join(path_us_tensors, name_tensor.replace(".npy",'_pixdim.npy')))).to(device=self.device)
        im_dim = torch.from_numpy(np.load(os.path.join(path_us_tensors, name_tensor.replace(".npy",'_imdim.npy')))).to(device=self.device)

        us_dict = {"image_dim":im_dim,
                   "voxel_size":0.5,
                   "pixel_size":pix_dim,
                   "volume":volume,
                   "origin":origin}
        self.us_dict = us_dict

    def forward(self):
        """
        Defines forward pass. Generate video and US data
        Pass through model.
        Re-render. Calculate loss.
        """
        # Base data
        # In PT frame of reference.
        transform_l2c = self.video_loader.l2c.expand(self.batch, -1, -1) # (N, 4, 4)
        transform_p2c = self.video_loader.p2c.expand(self.batch, -1, -1) # (N, 4, 4)

        # Base rendering objects - Video
        verts_liver = self.video_loader.meshes["liver"]["verts"].to(self.device) # (1, L, 3)
        faces_liver = self.video_loader.meshes["liver"]["faces"].to(self.device)# (1, G)
        textures_liver = self.video_loader.meshes["liver"]["textures"].to(self.device) # (1, L)
        verts_probe = self.video_loader.meshes["probe"]["verts"].to(self.device) # (1, P, 3)
        faces_probe = self.video_loader.meshes["probe"]["faces"].to(self.device) # (1, F)
        textures_probe = self.video_loader.meshes["probe"]["textures"].to(self.device) # (1, P) 
        verts_probe_batch = verts_probe.repeat(self.batch, 1, 1).to(self.device) # (N, P, 3)
        # Base rendering objects - US
        image_dim = self.us_dict["image_dim"]
        pixel_size = self.us_dict["pixel_size"]
        voxel_size = self.us_dict["voxel_size"]
        origin = self.us_dict["origin"]
        volume = self.us_dict["volume"]

        # Modify transforms p2c, l2c by perturbing them by some degree. For now we
        # use the original transform to demo.
        transform_p2c_perturbed = Transform3d(matrix=transform_p2c, device=self.device)
        transform_l2c_perturbed = Transform3d(matrix=transform_l2c, device=self.device)
        transform_c2l = transform_l2c_perturbed.inverse().to(self.device)

        # Transform l2c OpenGL/PyTorch into P2L slicesampler for US slicing.
        p2l_open_gl = transform_p2c_perturbed.compose(transform_c2l).get_matrix()
        r_p2l, t_p2l = vru.split_opengl_hom_matrix(p2l_open_gl.to(self.device))
        M_p2l_opencv, _, _ = vru.opengl_to_opencv(r_p2l.to(self.device), t_p2l.to(self.device), self.device)
        r_p2l_cv, t_p2l_cv = vru.split_opencv_hom_matrix(M_p2l_opencv)
        M_p2l_slicesampler = vru.p2l_2_slicesampler(M_p2l_opencv)

        # Get c2l in cv space
        r_c2l, t_c2l = vru.split_opengl_hom_matrix(transform_c2l.get_matrix().to(self.device))
        M_c2l_opencv, _, _ = vru.opengl_to_opencv(r_c2l.to(self.device), t_c2l.to(self.device), self.device)
        r_c2l_cv, t_c2l_cv = vru.split_opencv_hom_matrix(M_c2l_opencv)

        # Slice for camera rendering
        r_l2c, t_l2c = vru.split_opengl_hom_matrix(transform_l2c_perturbed.get_matrix())
        # Generate probe meshes
        verts_probe_unbatched = lvvmu.batch_verts_transformation(transform_c2l,
                                                                 transform_p2c_perturbed,
                                                                 verts_probe_batch,
                                                                 self.batch,
                                                                 self.device)
        # Create list of N meshes with both liver and probe.
        batch_meshes = lvvmu.generate_composite_probe_and_liver_meshes(verts_liver,
                                                                       faces_liver,
                                                                       textures_liver,
                                                                       verts_probe_unbatched,
                                                                       faces_probe,
                                                                       textures_probe,
                                                                       self.batch)
        # Normalise in openCV space
        t_p2l_norm = vru.global_to_local_space(t_p2l_cv, self.bounds)
        t_c2l_norm = vru.global_to_local_space(t_c2l_cv, self.bounds)

        #### RENDERING
        # Render image
        image = self.video_loader.renderer(meshes_world=batch_meshes, R=r_l2c, T=t_l2c) # (N, B, W, Ch)
        # Render US
        us = lvusg.slice_volume(image_dim,
                                pixel_size,
                                M_p2l_slicesampler,
                                voxel_size,
                                origin,
                                volume,
                                self.batch,
                                self.device)
        return image, us
