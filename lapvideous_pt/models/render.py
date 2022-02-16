# coding-utf-8

"""
Module that contains rendering procedures
for various objects in the scene
"""

import torch
import torch.nn.functional as F
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import rotation_conversions as p3drc
import kornia.morphology as m
# LapVideoUS-PT
import lapvideous_pt.generators.video_generation.utils as vru
import lapvideous_pt.generators.ultrasound_reslicing.us_generator as lvusg
import lapvideous_pt.generators.video_generation.mesh_utils as lvvmu
import lapvideous_pt.models.utils as mu
import lapvideous_pt.generators.augmentation.image_space_aug as lvisa

def get_mp2l_slicesampler(transform_l2c, transform_p2c, device):
    """
    Function to get slicesampler plane
    from input l2c, p2c.
    :param transform_l2c: Transform3d
    :param transform_p2c: Transform3d
    :param device: cuda device.
    """
    transform_c2l = transform_l2c.inverse()
    # Transform l2c OpenGL/PyTorch into P2L slicesampler for US slicing.
    p2l_open_gl = transform_p2c.compose(transform_c2l).get_matrix()
    r_p2l, t_p2l = vru.split_opengl_hom_matrix(p2l_open_gl.to(device))
    M_p2l_opencv, _, _ = vru.opengl_to_opencv_p2l(r_p2l.to(device), t_p2l.to(device), device)
    M_p2l_slicesampler = vru.p2l_2_slicesampler(M_p2l_opencv)
    return M_p2l_slicesampler, r_p2l, t_p2l

def render(batch_meshes, video_loader, r_l2c, t_l2c, video_noise):
    """
    For a given batch of mesh objects, use video_loader diff
    renderer and camera views defined by r_l2c, t_l2c
    to render images. Post process using video_noise, if
    passed.
    :param batch_meshes:
    :param video_loader:
    :param r_l2c:
    :param t_l2c:
    :param video_noise:
    :return: torch.Tensor
    """
    image = video_loader.renderer(meshes_world=batch_meshes, R=r_l2c, T=t_l2c) # (N, B, W, Ch)
    image = torch.transpose(image, 3, 1)
    if video_noise:
        if video_noise[0] is not None: # erosion
            image = m.erosion(image, video_noise[0])
        if video_noise[1] is not None: # dilation
            image = m.dilation(image, video_noise[1])
        if video_noise[2]: # dropout
            pass
    return image

def get_transformed_verts(liver_data,
                          transform_l2c,
                          transform_p2c,
                          batch,
                          device):
    """
    Method to transform homogenous vertices by given transformations.
    :param liver_data:
    :param transform_l2c: transform3d
    :param transform_p2c: transform3d
    :return:
        - verts_probe_unbatched
        - r_l2c, t_l2c
        - r_c2l, t_c2l
        - r_p2l, t_p2l
        - M_p2l_slicesampler

    """
    transform_c2l = transform_l2c.inverse()
    # We pass some p2c and l2c into the renderer.
    M_p2l_slicesampler, r_p2l, t_p2l = get_mp2l_slicesampler(transform_l2c,
                                                             transform_p2c,
                                                             device)

    # Get c2l
    r_c2l, t_c2l = vru.split_opengl_hom_matrix(transform_c2l.get_matrix().to(device))

    # Slice for camera rendering
    r_l2c, t_l2c = vru.split_opengl_hom_matrix(transform_l2c.get_matrix())
    # Generate probe meshes
    verts_probe_unbatched = lvvmu.batch_verts_transformation(transform_c2l,
                                                                transform_p2c,
                                                                liver_data[1][0],
                                                                batch,
                                                                device)
    return verts_probe_unbatched, [r_l2c, t_l2c], [r_c2l, t_c2l], [r_p2l, t_p2l], M_p2l_slicesampler

def render_vid(liver_data,
               video_loader,
               transform_l2c,
               transform_p2c=None,
               batch=5,
               video_noise=None,
               device=None):
    """
    Function that individually renders liver and probe
    mesh.
    :param liver_data:
    :param video_loader:
    :param transform_l2c: transform3d
    :param transform_p2c: transform3d
    :param batch: int
    :return: List[torch.Tensor]
    """
    verts_liver_unbatched = torch.split(liver_data[0][0],  [1 for i in range(batch)], 0)
    faces_liver_unbatched = torch.split(liver_data[0][1],  [1 for i in range(batch)], 0)    
    batch_meshes_liver = lvvmu.generate_object_mesh(verts_liver_unbatched, faces_liver_unbatched, liver_data[0][2], batch)
    r_l2c, t_l2c = vru.split_opengl_hom_matrix(transform_l2c.get_matrix())
    image_liver = render(batch_meshes_liver, video_loader, r_l2c, t_l2c, video_noise)

    if transform_p2c is not None:
        verts_probe_unbatched, _, _, _, _ =\
            get_transformed_verts(liver_data, transform_l2c, transform_p2c, batch, device)
        faces_probe_unbatched = torch.split(liver_data[1][1],  [1 for i in range(batch)], 0)
        batch_meshes_probe = lvvmu.generate_object_mesh(verts_probe_unbatched, faces_probe_unbatched, liver_data[1][2], batch)
        image_probe = render(batch_meshes_probe, video_loader, r_l2c, t_l2c, video_noise)
        return image_liver, image_probe

    return image_liver


def render_us(M_p2l_slicesampler,
              us_dict,
              us_dropout_params,
              us_pad,
              device,
              us_noise=None,
              batch=5):
    """
    Function to render ultrasound plane
    given a p2l matrix, us_dict of parameters,
    dropout/noise parameters, padding for the
    us image.
    :param M_p2l_slicesampler: torch.Tensor
    :param us_dict: dict
    :param us_dropout_params: dict
    :param us_pad: list
    :param device: cuda_device
    :param us_noise: List[list]
    :param batch: int
    """
    # Render US
    # Base rendering objects - US
    us = lvusg.slice_volume(us_dict["image_dim"],
                            us_dict["pixel_size"],
                            M_p2l_slicesampler,
                            us_dict["voxel_size"],
                            us_dict["origin"],
                            us_dict["volume"],
                            batch,
                            device)
    # Batch together, reshape US into correct output size.
    # B, ch, 1, 68, 83
    us = torch.squeeze(us, 2) # [N, ch, 68, 83]
    if us_noise:
        if us_noise[2] is not None: # dropout/delete vessels
            us = lvisa.delete_channel_features(us,
                                            us_noise[2],
                                            us_dropout_params["num_iterations"],
                                            us_dropout_params["num_features_del"],
                                            us_dropout_params["min_size_features"],
                                            us_dropout_params["max_size_features"])
        if us_noise[0] is not None: # erosion
            us = m.erosion(us, us_noise[0])
        if us_noise[1] is not None: # dilation
            us = m.erosion(us, us_noise[1])

    # Mask
    us = us * us_dict["us_mask"]
    us_pad = F.pad(us, us_pad) # (N, Ch, Out_size[0], Out_size[1])
    return us_pad

def render_scene(liver_data,
                 video_loader,
                 transform_l2c,
                 transform_p2c,
                 bounds,
                 us_dict,
                 us_dropout_params,
                 us_pad,
                 device,
                 us_noise=None,
                 video_noise=None,
                 alpha=None,
                 batch=5):
    """
    Render complete scene of combined probe and liver
    mesh, with US plane.
    :param liver_data:
    :param video_loader:
    :param transform_l2c: torch.Tensor
    :param transform_p2c: torch.Tensor
    :param bounds: torch.Tensor
    :param us_dict: dict
    :param us_dropout_params:
    :param us_pad:
    :param device: str indicating cuda device
    :param us_noise:
    :param video_noise:
    :param alpha: bool | None
    :param batch: int
    """
    l2c_pytorch3d = Transform3d(matrix=transform_l2c, device=device)
    p2c_pytorch3d = Transform3d(matrix=transform_p2c, device=device)
    verts_probe_unbatched, liver_transform, camera_transform, probe_transform, M_p2l_slicesampler =\
        get_transformed_verts(liver_data, l2c_pytorch3d, p2c_pytorch3d, batch, device)
    r_l2c, t_l2c = liver_transform
    r_c2l, t_c2l = camera_transform
    r_p2l, t_p2l = probe_transform
    verts_liver_unbatched = torch.split(liver_data[0][0],  [1 for i in range(batch)], 0)
    faces_liver_unbatched = torch.split(liver_data[0][1],  [1 for i in range(batch)], 0)
    faces_probe_unbatched = torch.split(liver_data[1][1],  [1 for i in range(batch)], 0)
    # Create list of N meshes with both liver and probe.
    batch_meshes = lvvmu.generate_composite_probe_and_liver_meshes(verts_liver_unbatched,
                                                                faces_liver_unbatched,
                                                                liver_data[0][2],
                                                                verts_probe_unbatched,
                                                                faces_probe_unbatched,
                                                                liver_data[1][2],
                                                                batch)
    # Normalise in openCV space
    t_p2l_norm = vru.global_to_local_space(t_p2l, bounds)
    t_c2l_norm = vru.global_to_local_space(t_c2l, bounds)

    #### RENDERING
    # Render image
    image = render(batch_meshes, video_loader, r_l2c, t_l2c, video_noise)

    # Render US
    us_pad = render_us(M_p2l_slicesampler,
                       us_dict,
                       us_dropout_params,
                       us_pad,
                       device,
                       us_noise,
                       batch)
    
    if alpha is None:
        # Get rid of the alpha channel in rendering.
        image = image[:, 0:3, :, :]
    image_tensor = torch.cat([image, torch.transpose(us_pad, 3, 2)], 1) # Cat along channels
    return image_tensor, [t_c2l_norm, t_p2l_norm]
