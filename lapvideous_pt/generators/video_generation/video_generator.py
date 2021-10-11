# coding=utf-8

"""
Pytorch differentiable video rendering generators
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import lapvideous_pt.generators.video_generation.utils as vgu
from pytorch3d.io import load_obj
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras,  
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    camera_conversions
)


class VideoLoader():
    """
    Class with useful utils to pre-process
    data prior to rendering.
    """
    def __init__(self,
                 mesh_dir,
                 config_dir,
                 liver2camera_reference,
                 probe2camera_reference,
                 intrinsics,
                 device):
        """
        :param mesh_dir: str, path to directory where meshes are stored.
        :param config_dir: str, path to .json file where 
        :param liver2camera_reference: str, path to spp_liver2camera.txt
        :param probe2camera_reference: str, path to spp_probe2camera.txt
        :param intrinsics: str, path to calibration matrix.
        :param device: torch.cuda device.
        """
        # Read config file.
        with open(config_dir) as f:
            config = json.load(f)

        self.mesh_dir = mesh_dir
        self.config_dir = config_dir
        self.config = config
        self.liver2camera_ref = liver2camera_reference
        self.probe2camera_ref = probe2camera_reference
        self.intrinsics = intrinsics
        self.device = device

    def pre_process_reference_poses(self, liver2camera_ref, probe2camera_ref):
        """
        Pre-process pre-simulated GT poses into PyTorch frame of reference.
        :param liver2camera_ref: str, path to liver2camera OpenCV reference pose.
        :param probe2camera_ref: str, path to probe2camera OpenCV reference pose.
        :return: void.
        """
        l2c_ref = np.loadtxt(liver2camera_ref)
        r_l2c = torch.from_numpy(np.expand_dims(l2c_ref[:3, :3], 0).astype(np.float32)).to(self.device) # [1, 3, 3]
        t_l2c = torch.from_numpy(np.expand_dims(l2c_ref[:3, 3], 0).astype(np.float32)).to(self.device) # [1, 3]
        p2c_ref = np.loadtxt(probe2camera_ref)
        r_p2c = torch.from_numpy(np.expand_dims(p2c_ref[:3, :3], 0).astype(np.float32)).to(self.device) # [1, 3, 3]
        t_p2c = torch.from_numpy(np.expand_dims(p2c_ref[:3, 3], 0).astype(np.float32)).to(self.device) # [1, 3]
        # Convert to PyTorch transforms.
        l2c_pytorch, _, _ = vgu.opencv_to_opengl(r_l2c, t_l2c, self.device)
        p2c_pytorch, _, _ = vgu.opencv_to_opengl(r_p2c, t_p2c, self.device)
        # Store
        self.l2c = l2c_pytorch
        self.p2c = p2c_pytorch

    def load_meshes(self, mesh_dir, config):
        """
        Load meshes and generate meshes objects
        :param mesh_dir: str, path where meshes are stored.
        :param config: dict, that contains keys with object names
                       and paths to the corresponding .obj file.
        :return: void.
        """
        meshes = {}
        for object in config.keys():
            meshes[object] = {}
            # Load the mesh
            verts, faces_idx, _ = load_obj(os.path.join(mesh_dir, config[object]["path"]))
            faces = faces_idx.verts_idx
            # For now just use RGB values with flat shading.]
            colour = np.array(config[object]["colour"]) # list (3) of 0-255 values for colour.
            verts_rgb = np.ones_like(verts) 
            verts_rgb *= colour
            verts_rgb = torch.from_numpy(verts_rgb).expand(1, -1, -1) # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
            # Add to the dict
            meshes[object]["verts"] = verts
            meshes[object]["faces"] = faces
            meshes[object]["textures"] = textures

        self.meshes = meshes

    def setup_renderer(self, intrinsics, image_size, output_size):
        """
        Sets up a Perspective camera from a series of OpenCV
        parameters
        :param intrinsics: np.array of camera intrinsics. OpenCV
                           style.
        :param image_size: list, (2,). Camera dimensions for original
                           calibration matrix.
        :param output_size: list, (2,). Image dimensions output.
                            PyTorch3d rescales the data to this
                            output size.
        :return: void
        """
        # Use one set of intrinsics only
        K = torch.from_numpy(np.expand_dims(intrinsics, 0).astype(np.float32)).to(self.device) # 1, 3, 3
        WH = torch.from_numpy(np.expand_dims(np.array(image_size), 0).astype(np.float32)).to(self.device)

        # Initialize a perspective camera - we do this using a conversion from OpenCV.
        # The initial location of the camera doesn't matter as we can set this individually
        # each time later. So just set identity matrix.
        eye =  torch.from_numpy(np.expand_dims(np.eye(3), 0).astype(np.float32)).to(self.device)
        t_eye =  torch.from_numpy(np.expand_dims(np.zeros(3), 0).astype(np.float32)).to(self.device)
        cameras = camera_conversions._cameras_from_opencv_projection(eye, t_eye, K, WH)

        # Generate params and settings for Phong renderer.
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=output_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
        )
        rasterizer = MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings)

        renderer_camera_location = rasterizer.cameras.get_camera_center()

        # We can add a point light in front of the object. 
        lights = PointLights(device=self.device, location=(renderer_camera_location))
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=self.device, cameras=cameras, lights=lights)
        )
        self.renderer = phong_renderer
        self.renderer.to(self.device)
