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
import kornia.morphology as m
# LapVideoUS-PT
import lapvideous_pt.generators.video_generation.utils as vru
import lapvideous_pt.generators.ultrasound_reslicing.us_generator as lvusg
from lapvideous_pt.generators.video_generation.video_generator import VideoLoader
import lapvideous_pt.generators.video_generation.mesh_utils as lvvmu
import lapvideous_pt.models.utils as mu
import lapvideous_pt.generators.augmentation.image_space_aug as lvisa
import lapvideous_pt.models.render as lvrender

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
                 device,
                 model_config_dict=None,
                 mask_path=None,
                 alpha=None,
                 loss_dict=None):
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
        :param model_config_dict: dict, of parameters to build nn
        :param mask_path: str, default=None
        :param alpha: bool, default=None, defines whether or not alpha channel
                      in renderer is being used.
        :param loss_dict: List[float], default=None, defines loss behaviour.
        """
        super().__init__()
        # Setup CUDA device.
        if device == "cluster":
            if torch.cuda.is_available():
                device = "cuda:0"
                print("Using CUDA Device: ", torch.device(device))
                device = torch.device("cuda:0")
        elif not device=="cpu":
            if torch.cuda.is_available():
                print("Using CUDA Device: ", torch.device(device))
                device = torch.device(device)
        else:
            device = torch.device("cpu")

        self.meshes_dict = mesh_dir
        self.device = device
        self.batch = batch
        self.image_size = image_size
        self.output_size = output_size
        self.alpha=alpha
        print("Mem allocated before video: ", torch.cuda.memory_allocated())
        self.pre_process_video_files(mesh_dir,
                                     config_dir,
                                     liver2camera_reference,
                                     probe2camera_reference,
                                     image_size,
                                     output_size,
                                     intrinsics,
                                     device)
        if mask_path is None:
            self.mask_path = os.path.join(mesh_dir, "us_mask.npy")
        else:
            self.mask_path = mask_path
        if loss_dict is None:
            self.loss_dict_weights = {"c2l_r":100.0,
                                      "p2l_r":100.0,
                                      "c2l_t":1/100.0,
                                      "p2l_t":1/100.0,
                                      "im_ch_1":1/100.0,
                                      "im_ch_2":1/100.0,
                                      "im_ch_3":1/100.0}
        else:
            self.loss_dict_weights = loss_dict

        print("Mem allocated after video: ", torch.cuda.memory_allocated())
        print("Mem allocated before US: ", torch.cuda.memory_allocated())
        self.pre_process_US_files(path_us_tensors,
                                  name_tensor,
                                  self.mask_path)
        # Default no us augmentation
        self.us_dropout_params = {"channel_ops":None,
                                  "proba_dropout":None,
                                  "num_iterations":None,
                                  "num_features_del":None,
                                  "min_size_features":None,
                                  "max_size_features":None
                                 }
        print("Mem allocated after US: ", torch.cuda.memory_allocated())
        print("Mem allocated before model build: ", torch.cuda.memory_allocated())
        self.build_nn(output_size, model_config_dict)
        self.to(device).float()
        print("Mem allocated after model build: ", torch.cuda.memory_allocated())


    def build_nn(self, image_size, model_config_dict):
        """
        Class method to build neural network.
        Simple couple of convolutional layers with
        FCNs at the end.
        :param image_size:
        """
        print("Building Model...")
        self.conv_backbone, self.conv_backbone_names = mu.parse_model_config_convs(model_config_dict["conv_backbone"])
        self.branch1, self.branch1_names = mu.parse_model_config_linear(model_config_dict["branch1"])
        self.branch2, self.branch2_names = mu.parse_model_config_linear(model_config_dict["branch2"])
        self.fc1 = nn.Linear(**model_config_dict["fc1_layer"])
        self.split_size = model_config_dict["split_size"]
        self.branch_split = model_config_dict["branch_split"]
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
        self.bounds = torch.from_numpy(np.array([500.0, 500.0, 500.0], dtype=np.float32)).to(self.device)

    def define_dropout_params(self,
                              channel_ops,
                              num_iterations,
                              num_features_del,
                              min_size_features,
                              max_size_features):
        """
        Method to modify the US dropout params, which
        by default are constructed as all False.

        :param channel_ops: List[Bool]
        :param num_iterations: int,
        :param num_features_del: List[int]
        :param min_size_feaures: List[int]
        :param max_size_features: List[int]
        """
        # Check all lists are the same size
        assert len({len(i) for i in [channel_ops,
                                     num_iterations,
                                     num_features_del,
                                     min_size_features,
                                     max_size_features]}) == 1
        self.us_dropout_params = {"channel_ops":channel_ops,
                                  "num_iterations":num_iterations,
                                  "num_features_del":num_features_del,
                                  "min_size_features":min_size_features,
                                  "max_size_features":max_size_features
                                 }

    def pre_process_US_files(self, path_us_tensors, name_tensor, mask_path):
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
        origin = torch.from_numpy(np.load(os.path.join(path_us_tensors, name_tensor.replace(".npy", "_origin.npy")))).float().to(device=self.device)
        pix_dim = torch.from_numpy(np.load(os.path.join(path_us_tensors, name_tensor.replace(".npy",'_pixdim.npy')))).float().to(device=self.device)
        im_dim = torch.from_numpy(np.load(os.path.join(path_us_tensors, name_tensor.replace(".npy",'_imdim.npy')))).int().to(device=self.device)
        us_mask = torch.where(torch.from_numpy(np.transpose(np.load(mask_path), [3, 2, 0, 1])).float().to(device=self.device)>0,
                              torch.ones(1).float().to(self.device),
                              torch.zeros(1).float().to(self.device))
        print("Im dim US: ", im_dim)
        us_dict = {"image_dim":im_dim,
                   "voxel_size":0.5,
                   "pixel_size":pix_dim,
                   "volume":volume,
                   "origin":origin,
                   "us_mask":us_mask}
        # Calculate us diff for padding
        diff_us_size = list(self.output_size - im_dim.cpu().numpy()[0:2])
        list_padding = []
        for item in diff_us_size:
            if item %2 != 0:
                list_padding.extend([int(np.floor(item/2)), int(np.ceil(item/2))])
            else:
                list_padding.extend([int(item/2), int(item/2)])
        print(tuple(list_padding))
        # self.us_pad = tuple([list_padding[2], list_padding[3], list_padding[0], list_padding[1]])
        self.us_pad = tuple(list_padding)
        self.us_dict = us_dict

    def prep_input_data_for_render(self):
        """
        Get mesh data and pre-process it for
        rendering. This way we only generate one batch
        of video rendering objects once, and avoid
        re-calling it each rendering instance.
        :return: [liver_verts, liver_faces, liver_textures],
                 [probe_verts, probe_faces, probe_textures],
                 us_volume
        """
        # Base rendering objects - Video
        verts_liver = self.video_loader.meshes["liver"]["verts"].float().to(self.device) # (1, L, 3)
        faces_liver = self.video_loader.meshes["liver"]["faces"].float().to(self.device)# (1, G)
        textures_liver = self.video_loader.meshes["liver"]["textures"].to(self.device) # (1, L)
        verts_probe = self.video_loader.meshes["probe"]["verts"].float().to(self.device) # (1, P, 3)
        faces_probe = self.video_loader.meshes["probe"]["faces"].float().to(self.device) # (1, F)
        textures_probe = self.video_loader.meshes["probe"]["textures"].to(self.device) # (1, P)
        batch_textures_liver = [textures_liver for i in range(self.batch)]
        batch_textures_probe = [textures_probe for i in range(self.batch)]
        batch_faces_probe = faces_probe.repeat(self.batch, 1, 1).to(self.device)
        batch_faces_liver = faces_liver.repeat(self.batch, 1, 1).to(self.device)
        batch_faces_probe = faces_probe.repeat(self.batch, 1, 1).to(self.device)
        verts_liver_batch = verts_liver.repeat(self.batch, 1, 1).to(self.device) # (N, P, 3)
        verts_probe_batch = verts_probe.repeat(self.batch, 1, 1).to(self.device) # (N, P, 3)
        # Prep the US volume
        return [[verts_liver_batch, batch_faces_liver, batch_textures_liver], \
               [verts_probe_batch, batch_faces_probe, batch_textures_probe]]

    def get_transformed_verts(self,
                              liver_data,
                              transform_l2c,
                              transform_p2c):
        """
        Class method to transform homogenous vertices
        by given transformations.
        :param liver_data:
        :param transform_l2c: torch.Tensor
        :param transform_p2c: torch.Tensor
        :return:
            - verts_probe_unbatched
            - r_l2c, t_l2c
            - r_c2l, t_c2l
            - r_p2l, t_p2l
            - M_p2l_slicesampler

        """
        outputs = lvrender.get_transformed_verts(liver_data,
                                                 transform_l2c,
                                                 transform_p2c,
                                                 self.batch,
                                                 self.device)
        return outputs

    def render_data(self,
                    liver_data,
                    transform_l2c,
                    transform_p2c,
                    us_noise=None,
                    video_noise=None):
        """
        Generate some image data based on a given set
        of transforms and rendering data tensors.
        :param liver_data: List[List[torch.Tensor]], (2,), data for differentiable
                            video rendering:
                            - [verts_liver_batch, batch_faces_liver, batch_textures_liver]
                            - [verts_probe_batch, batch_faces_probe, batch_textures_probe]
        :param us_volume: torch.Tensor, (N, ...)
        :param transform_p2c: torch.Tensor, [N, 4, 4]
        :param transform_l2c: torch.Tensor, [N, 4, 4]
        :param us_noise: List[torch.Tensor or None], (default=None), (3,), [erosion, dilation, dropout]
        :param video_noise: List[torch.Tensor or None], (default=None), (3,), [erosion, dilation, dropout]
        :return: torch.Tensor for image and video data.
                    - (N, Ch_vid, H_vid, W_vid), 0:4 video, 4:7 US.
        """
        image_tensor, labels = lvrender.render_scene(liver_data,
                                        self.video_loader,
                                        transform_l2c,
                                        transform_p2c,
                                        self.bounds,
                                        self.us_dict,
                                        self.us_dropout_params,
                                        self.us_pad,
                                        self.device,
                                        us_noise,
                                        video_noise,
                                        self.alpha,
                                        self.batch)
        return image_tensor, labels

    def post_process_predictions(self, quat, transl,):
        """
        Class method that defines how the data from predictions
        is post processed. We assume in this case that the rotations
        predicted are quaternions and
        also refer to rotations that
        are in Pytorch3d frame of reference already,
        so we can just return the Transform3d to pass directly to the
        renderer.
        :param c2l_q: torch.Tensor, c2l rotation predictions (B, 4,)
        :param c2l_t: torch.Tensor, c2l translation predictions (B, 3,)
        :param p2l_q: torch.Tensor, p2l rotation predictions (B, 4,)
        :param p2l_t: torch.Tensor, p2l translation predictions (B, 3,)
        :return: List[Transform3d], (2,), c2l and p2l
        """
        # Convert quats to rot matrices
        rot = p3drc.quaternion_to_matrix(quat)
        transl_g = vru.local_to_global_space(transl, self.bounds)
        M_opengGL = vru.cat_opengl_hom_matrix(torch.transpose(rot, 2, 1), transl_g, self.device)
        # Return Transform3d
        M_pytorch3d = Transform3d(matrix=M_opengGL, device=self.device)
        return M_pytorch3d

    def forward(self, liver_data, image_data):
        """
        Defines forward pass.
        Pass data through model.
        Re-render calculated poses.
        Calculate loss.
        :param liver_data: List[List[torch.Tensor]], (2,), data for differentiable
                           video rendering:
                          - [verts_liver_batch, batch_faces_liver, batch_textures_liver]
                          - [verts_probe_batch, batch_faces_probe, batch_textures_probe]
        :param us_volume: torch.Tensor, (N, W, H, D,Ch)
        :param data: torch.Tensor, (N, Ch, H, W),
                     image from GT rendering.
        :return: [loss_im, tensor_pred, norm_trans]
        """
        #### NETWORKS
        # Backbone
        out = image_data
        for i, layer in enumerate(self.conv_backbone):
            if self.conv_backbone_names[i] == "Conv2d":
                out = F.leaky_relu(layer(out))
            else: # Maxpool or BatchNorm
                out = layer(out)

        # Last conv layer, flatten and leaky relu
        out = torch.flatten(out, start_dim=1)
        out = F.leaky_relu(self.fc1(out))
        # Split network results into two branches
        out_branch_1, out_branch_2 = torch.split(out, [self.split_size, self.split_size], dim=1)
        
        for i, layer in enumerate(self.branch1[:-1]):
            if self.branch1_names[i] == "Linear":
                out_branch_1 = F.leaky_relu(layer(out_branch_1))
            else:
                out_branch_1 = layer(out_branch_1)
        # Last layer without an activation.
        out_branch_1 = self.branch1[-1](out_branch_1)

        for i, layer in enumerate(self.branch2[:-1]):
            if self.branch2_names[i] == "Linear":
                out_branch_2 = F.leaky_relu(layer(out_branch_2))
            else:
                out_branch_2 = layer(out_branch_2)

        # Last layer without an activation.
        out_branch_2 = self.branch2[-1](out_branch_2)


        #### POST PROCESS OUTPUTS - depending on architecture,
        # split accordingly.
        if self.branch_split == "p2l_c2l":
            # Branches are 7 dimensional.
            c2l_q, c2l_t = torch.split(out_branch_1, [4, 3], dim=1)
            p2l_q, p2l_t = torch.split(out_branch_2, [4, 3], dim=1)
        else:
            # Branches are 8 and 6 dimensional respectively.
            c2l_q, p2l_q = torch.split(out_branch_1, [4, 4], dim=1)
            c2l_t, p2l_t = torch.split(out_branch_2, [3, 3], dim=1)

        # For translations we need a representation in bounded space
        c2l_pytorch3d = self.post_process_predictions(quat=c2l_q,
                                                      transl=c2l_t)
        p2l_pytorch3d = self.post_process_predictions(quat=p2l_q,
                                                      transl=p2l_t)

        transform_l2c_pred = c2l_pytorch3d.inverse()
        transform_p2c_pred = p2l_pytorch3d.compose(transform_l2c_pred)
        tensor_pred = self.render_data(liver_data=liver_data,
                                          transform_p2c=transform_p2c_pred.get_matrix(),
                                          transform_l2c=transform_l2c_pred.get_matrix())
        #### RETURN PRED IMAGE AND PRED TRANSFORMS
        return tensor_pred, [c2l_pytorch3d, p2l_pytorch3d], torch.cat((c2l_t, p2l_t), dim=1)

class LapVideo(LapVideoUS):
    """
    Render video features separately
    """
    def render_data(self,
                    liver_data,
                    transform_l2c,
                    video_noise=None):
        """
        """
        image_liver = lvrender.render_vid(liver_data,
                                          self.video_loader,
                                          transform_l2c,
                                          transform_p2c=None,
                                          batch=self.batch,
                                          video_noise=video_noise,
                                          device=self.device)
        return image_liver

    def build_nn(self, image_size, model_config_dict):
        """
        Class method to build neural network.
        Simple couple of convolutional layers with
        FCNs at the end.
        :param image_size:
        """
        print("Building Model...")
        self.conv_backbone, self.conv_backbone_names = mu.parse_model_config_convs(model_config_dict["conv_backbone"])
        self.branch1, self.branch1_names = mu.parse_model_config_linear(model_config_dict["branch1"])
        self.fc1 = nn.Linear(**model_config_dict["fc1_layer"])
        print("Model built.")

    def forward(self, liver_data, image_data):
        """
        Defines forward pass.
        Pass data through model.
        Re-render calculated poses.
        Calculate loss.
        :param liver_data: List[List[torch.Tensor]], (2,), data for differentiable
                           video rendering:
                          - [verts_liver_batch, batch_faces_liver, batch_textures_liver]
                          - [verts_probe_batch, batch_faces_probe, batch_textures_probe]
        :param us_volume: torch.Tensor, (N, W, H, D,Ch)
        :param data: torch.Tensor, (N, Ch, H, W),
                     image from GT rendering.
        :return: [loss_im, tensor_pred, norm_trans]
        """
        #### NETWORKS
        # Backbone
        out = image_data
        for i, layer in enumerate(self.conv_backbone):
            if self.conv_backbone_names[i] == "Conv2d":
                out = F.leaky_relu(layer(out))
            else: # Maxpool or BatchNorm
                out = layer(out)

        # Last conv layer, flatten and leaky relu
        out = torch.flatten(out, start_dim=1)
        out_branch_1 = F.leaky_relu(self.fc1(out))
        # One branch
        for i, layer in enumerate(self.branch1[:-1]):
            if self.branch1_names[i] == "Linear":
                out_branch_1 = F.leaky_relu(layer(out_branch_1))
            else:
                out_branch_1 = layer(out_branch_1)
        # Last layer without an activation.
        out_branch_1 = self.branch1[-1](out_branch_1)

        #### POST PROCESS OUTPUTS - depending on architecture,
        # split accordingly.
        c2l_q, c2l_t = torch.split(out_branch_1, [4, 3], dim=1)

        # For translations we need a representation in bounded space
        # For translations we need a representation in bounded space
        c2l_pytorch3d = self.post_process_predictions(quat=c2l_q,
                                                      transl=c2l_t)

        transform_l2c_pred = c2l_pytorch3d.inverse()
        tensor_pred = self.render_data(liver_data=liver_data,
                                       transform_l2c=transform_l2c_pred)
        #### RETURN PRED IMAGE AND PRED TRANSFORMS
        return tensor_pred, c2l_pytorch3d

class LapUS(LapVideoUS):
    """
    Just render US.
    """
    def render_data(self,
                    transform_p2l,
                    us_noise=None):
        """
        """
        r_p2l, t_p2l = vru.split_opengl_hom_matrix(transform_p2l.to(self.device))
        M_p2l_opencv, _, _ = vru.opengl_to_opencv_p2l(r_p2l.to(self.device), t_p2l.to(self.device), self.device)
        M_p2l_slicesampler = vru.p2l_2_slicesampler(M_p2l_opencv)
        image_us = lvrender.render_us(M_p2l_slicesampler,
                                      self.us_dict,
                                      self.us_dropout_params,
                                      self.us_pad,
                                      self.device,
                                      us_noise,
                                      self.batch)
        return image_us

    def build_nn(self, image_size, model_config_dict):
        """
        Class method to build neural network.
        Simple couple of convolutional layers with
        FCNs at the end.
        :param image_size:
        """
        print("Building Model...")
        self.conv_backbone, self.conv_backbone_names = mu.parse_model_config_convs(model_config_dict["conv_backbone"])
        self.branch1, self.branch1_names = mu.parse_model_config_linear(model_config_dict["branch1"])
        self.fc1 = nn.Linear(**model_config_dict["fc1_layer"])
        print("Model built.")

    def forward(self, image_data):
        """
        Defines forward pass.
        Pass data through model.
        Re-render calculated poses.
        Calculate loss.
        :param liver_data: List[List[torch.Tensor]], (2,), data for differentiable
                           video rendering:
                          - [verts_liver_batch, batch_faces_liver, batch_textures_liver]
                          - [verts_probe_batch, batch_faces_probe, batch_textures_probe]
        :param us_volume: torch.Tensor, (N, W, H, D,Ch)
        :param data: torch.Tensor, (N, Ch, H, W),
                     image from GT rendering.
        :return: [loss_im, tensor_pred, norm_trans]
        """
        #### NETWORKS
        # Backbone
        out = image_data
        for i, layer in enumerate(self.conv_backbone):
            if self.conv_backbone_names[i] == "Conv2d":
                out = F.leaky_relu(layer(out))
            else: # Maxpool or BatchNorm
                out = layer(out)

        # Last conv layer, flatten and leaky relu
        out = torch.flatten(out, start_dim=1)
        out_branch_1 = F.leaky_relu(self.fc1(out))
        # One branch
        for i, layer in enumerate(self.branch1[:-1]):
            if self.branch1_names[i] == "Linear":
                out_branch_1 = F.leaky_relu(layer(out_branch_1))
            else:
                out_branch_1 = layer(out_branch_1)
        # Last layer without an activation.
        out_branch_1 = self.branch1[-1](out_branch_1)

        #### POST PROCESS OUTPUTS - depending on architecture,
        # split accordingly.
        p2l_q, p2l_t = torch.split(out_branch_1, [4, 3], dim=1)

        p2l_pytorch3d = self.post_process_predictions(quat=p2l_q,
                                                      transl=p2l_t)

        tensor_pred = self.render_data(transform_p2l=p2l_pytorch3d.get_matrix())
        #### RETURN PRED IMAGE AND PRED TRANSFORMS
        return tensor_pred, p2l_pytorch3d

class LapVideoUSSeparate(LapVideoUS):
    """
    Just render all features independently.
    """
    def render_data(self,
                    liver_data,
                    transform_l2c,
                    transform_p2c,
                    us_noise=None,
                    video_noise=None):
        """
        """
        transform_p2c_perturbed = Transform3d(matrix=transform_p2c, device=self.device)
        transform_l2c_perturbed = Transform3d(matrix=transform_l2c, device=self.device)
        M_p2l_slicesampler, _, _ = lvrender.get_mp2l_slicesampler(transform_l2c_perturbed,
                                                                  transform_p2c_perturbed,
                                                                  self.device)

        image_us = lvrender.render_us(M_p2l_slicesampler,
                                      self.us_dict,
                                      self.us_dropout_params,
                                      self.us_pad,
                                      self.device,
                                      us_noise,
                                      self.batch)
        image_liver, image_probe = lvrender.render_vid(liver_data,
                                                       self.video_loader,
                                                       transform_l2c_perturbed,
                                                       transform_p2c_perturbed,
                                                       self.batch,
                                                       video_noise,
                                                       self.device)
        return image_liver, image_probe, image_us
