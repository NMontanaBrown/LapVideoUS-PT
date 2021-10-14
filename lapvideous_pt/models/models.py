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
import torch.nn.functional as F
from vtk.numpy_interface.dataset_adapter import NoneArray
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import rotation_conversions as p3drc
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
            print("Using CUDA Device: ", torch.device(device))
            device = torch.device(device)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        self.meshes_dict = mesh_dir
        self.device = device
        self.batch = batch
        self.image_size = image_size
        print("Mem allocated before video: ", torch.cuda.memory_allocated())
        self.pre_process_video_files(mesh_dir,
                                     config_dir,
                                     liver2camera_reference,
                                     probe2camera_reference,
                                     image_size,
                                     output_size,
                                     intrinsics,
                                     device)
        print("Mem allocated after video: ", torch.cuda.memory_allocated())
        print("Mem allocated before US: ", torch.cuda.memory_allocated())
        self.pre_process_US_files(path_us_tensors,
                                  name_tensor)
        print("Mem allocated after US: ", torch.cuda.memory_allocated())
        print("Mem allocated before model build: ", torch.cuda.memory_allocated())
        self.build_nn(output_size)
        self.to(device).float()
        print("Mem allocated after model build: ", torch.cuda.memory_allocated())


    def build_nn(self, image_size):
        """
        Class method to build neural network.
        Simple couple of convolutional layers with
        FCNs at the end.
        :param image_size:
        """
        print("Building Model...")
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=3, stride=1, padding=1) # Hout = Hin
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1) # Hout = Hin
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=2) # Hout/2
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2,2) # Hout/8
        self.fc1 = nn.Linear(30000, 10000)
        self.fc2 = nn.Linear(10000, 5000)
        self.fc3 = nn.Linear(5000, 1000)
        self.fc4 = nn.Linear(1000, 100)
        self.fc5 = nn.Linear(100, 14)
        print("Model built.")

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
        :param mesh_dir:
        :param config_dir:
        :param liver2camera_reference:
        :param prob2camera_reference:
        :param image_size:
        :param output_size:
        :param intrinsics:
        :param device:
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

    def render_data(self, transform_l2c, transform_p2c):
        """
        Generate some data based on a given set
        of transforms.
        :param transform_p2c: torch.Tensor, [N, 4, 4]
        :param transform_l2c: torch.Tensor, [N, 4, 4]
        :return: [image, us], torch.Tensors
        """
        # Base rendering objects - Video
        verts_liver = self.video_loader.meshes["liver"]["verts"].to(self.device) # (1, L, 3)
        faces_liver = self.video_loader.meshes["liver"]["faces"].to(self.device)# (1, G)
        textures_liver = self.video_loader.meshes["liver"]["textures"].to(self.device) # (1, L)
        verts_probe = self.video_loader.meshes["probe"]["verts"].to(self.device) # (1, P, 3)
        faces_probe = self.video_loader.meshes["probe"]["faces"].to(self.device) # (1, F)
        textures_probe = self.video_loader.meshes["probe"]["textures"].to(self.device) # (1, P)
        batch_textures_liver = [textures_liver for i in range(self.batch)]
        batch_textures_probe = [textures_probe for i in range(self.batch)]
        batch_faces_probe = faces_probe.repeat(self.batch, 1, 1).to(self.device)
        batch_faces_liver = faces_liver.repeat(self.batch, 1, 1).to(self.device)
        batch_faces_probe = faces_probe.repeat(self.batch, 1, 1).to(self.device)
        verts_liver_batch = verts_liver.repeat(self.batch, 1, 1).to(self.device) # (N, P, 3)
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
        M_p2l_opencv, _, _ = vru.opengl_to_opencv_p2l(r_p2l.to(self.device), t_p2l.to(self.device), self.device)
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
        verts_liver_unbatched = torch.split(verts_liver_batch,  [1 for i in range(self.batch)], 0)
        faces_liver_unbatched = torch.split(batch_faces_liver,  [1 for i in range(self.batch)], 0)
        faces_probe_unbatched = torch.split(batch_faces_probe,  [1 for i in range(self.batch)], 0)
        # Create list of N meshes with both liver and probe.
        batch_meshes = lvvmu.generate_composite_probe_and_liver_meshes(verts_liver_unbatched,
                                                                       faces_liver_unbatched,
                                                                       batch_textures_liver,
                                                                       verts_probe_unbatched,
                                                                       faces_probe_unbatched,
                                                                       batch_textures_probe,
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

    def forward(self, data):
        """
        Defines forward pass.
        Pass data through model.
        Re-render calculated poses.
        Calculate loss.
        :param data: torch.Tensor, (N, Ch, H, W)
        :return: [loss_im, image_pred, us_pred]
        """

        #### NETWORKS
        out = F.relu(self.conv1(data))
        out = F.relu(self.conv2(out))
        out = F.relu(self.pool(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.pool2(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = F.relu(self.pool3(out))
        out = torch.flatten(out, start_dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.tanh(self.fc5(out))

        #### POST PROCESS OUTPUTS - 14 vector of [-1, 1]
        # For translations we need a representation in bounded space
        c2l_q, c2l_t, p2l_q, p2l_t = torch.split(out, [4, 3, 4, 3], dim=1)
        c2l_r = p3drc.quaternion_to_matrix(c2l_q)
        p2l_r = p3drc.quaternion_to_matrix(p2l_q)
        c2l_t_global = vru.local_to_global_space(c2l_t, self.bounds)
        p2l_t_global = vru.local_to_global_space(p2l_t, self.bounds)
        p2l_opengGL, _, _ = vru.opencv_to_opengl_p2l(p2l_r, p2l_t_global, self.device)
        c2l_openGL, _, _ = vru.opencv_to_opengl(c2l_r, c2l_t_global, self.device)

        p2l_pytorch3d = Transform3d(matrix=p2l_opengGL, device=self.device)
        c2l_pytorch3d = Transform3d(matrix=c2l_openGL, device=self.device)

        transform_l2c_pred = c2l_pytorch3d.inverse()
        transform_p2c_pred = p2l_pytorch3d.compose(transform_l2c_pred)
        image_pred, us_pred = self.render_data(transform_p2c=transform_p2c_pred.get_matrix(),
                                               transform_l2c=transform_l2c_pred.get_matrix())

        #### CALCULATE LOSS
        loss_us = torch.nn.MSELoss()
        # loss_u = loss_us(data, us_pred)
        loss_im = loss_us(data, torch.transpose(image_pred, 3, 1))
        return loss_im, image_pred, us_pred
