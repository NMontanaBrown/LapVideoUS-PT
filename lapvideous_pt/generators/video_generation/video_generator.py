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
    # camera_conversions
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
            colour = np.array(config[object]["colour"], dtype=np.float32) # list (3) of 0-255 values for colour.
            verts_rgb = np.ones_like(verts, dtype=np.float32)
            verts_rgb *= colour
            verts_rgb = torch.from_numpy(verts_rgb).expand(1, -1, -1) # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
            # Add to the dict
            meshes[object]["verts"] = verts
            meshes[object]["faces"] = faces
            meshes[object]["textures"] = textures

        self.meshes = meshes

    def setup_renderer(self,
                       intrinsics=(1892.33, 944.889),
                       principal_point=(1105.87, 752.693),
                       image_size=(1920.0, 1080.0),
                       output_size=200):
        """
        Sets up a Perspective camera from a series of OpenCV
        parameters
        :param intrinsics: tuple,
        :param principal_point: tuple,
        :param image_size: tuple,
        :param output_size: list, (2,) or int. Image dimensions output.
                            PyTorch3d rescales the data to this
                            output size.
        :return: void
        """
        # Generate params and settings for Phong renderer.
        # Hard code faces per bin - otherwise we can get patches
        # missing from the pictures rendered. See issue #1 for details.
        self.cameras = PerspectiveCameras(
            focal_length=(intrinsics,),
            principal_point=(principal_point,),
            image_size=(image_size,),
            device=self.device)

        # Create a phong mesh renderer.
        # Set the background colour black.
        blend_params = BlendParams(background_color=(0, 0, 0))

        raster_settings = RasterizationSettings(
            image_size=output_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=100000,
            cull_backfaces=True
        )

        # Light is at the camera.
        self.lights_phong = \
            PointLights(device=self.device,
                        # diffuse_color=((0, 0, 0),),
                        # specular_color=((0, 0, 0),),
                        # ambient_color=((1.0, 1.0, 1.0),),))
            )

        self.mesh_rasteriser = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=raster_settings)

        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=self.device,
                                   cameras=self.cameras,
                                   lights=self.lights_phong,
                                   blend_params=blend_params)
        )

        self.renderer = self.phong_renderer
        self.renderer.to(self.device)
