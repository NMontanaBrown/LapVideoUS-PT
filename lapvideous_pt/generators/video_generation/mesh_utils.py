# coding=utf-8

"""
Utils to do with mesh batching and vert transformation
for rendering of liver and probe.
"""

import torch
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch

def batch_verts_transformation(c2l, p2c, verts_probe, batch, device):
    """
    Creates a batch of transformed probe vertices in liver
    reference (p2l).
    :param c2l: pytorch3d.Transform3d, camera2liver in pytorch space. (N, 4, 4)
    :param p2c: pytorch3d.Transform3d, probe2camera in pytorch space. (N, 4, 4)
    :param verts_probe: torch.Tensor, (N, B, 3), batch of probe vertices.
    :param batch: int, (N), size of batch.
    :return: verts_probe_unbatched, List[torch.Tensor], Nx (1, B, 3) 
    """
    verts_probe_transformed = p2c.compose(c2l).transform_points(verts_probe) # (N, P_probe, 3)
    verts_probe_unbatched = torch.split(verts_probe_transformed,  [1 for i in range(batch)], 0)
    return verts_probe_unbatched

def generate_composite_probe_and_liver_meshes(verts_liver_unbatched,
                                              faces_liver,
                                              textures_liver,
                                              verts_probe_unbatched,
                                              faces_probe,
                                              textures_probe,
                                              batch):
    """
    Generates meshes for rendering by combining a liver mesh
    with a probe mesh in an ordered fashion such that each batch
    represents the scene of one liver2camera and probe2liver
    transformation.
    :param verts_liver_unbatched: List[torch.Tensor], (N,), vertices
                        of liver mesh.
    :param faces_liver: List[torch.Tensor], Nx(F,), faces of liver mesh. 
    :param textures_liver: 
    :param verts_probe_unbatched: List[torch.Tensor], (N,),
                                  transformed vertices in liver space.
    :param faces_probe: List[torch.Tensor] Nx(G,), faces of probe mesh.
    :param textures_probe:
    :param batch: int, (N), batch size.
    :return: List[Meshes], (N,)
    """
    scene_meshes = [
            join_meshes_as_scene([Meshes(verts=verts_liver_unbatched[i],
                                         faces=faces_liver[i],
                                         textures=textures_liver[i]),
                                 Meshes(verts=verts_probe_unbatched[i],
                                        faces=faces_probe[i],
                                        textures=textures_probe[i])]) for i in range(batch)]
    batch_meshes = join_meshes_as_batch(scene_meshes)
    return batch_meshes

def verts_transformed_from_preds(c2l, p2c, verts_probe, verts_liver):
    """
    Creates a batch of transformed probe vertices in liver
    reference (p2l).
    :param c2l: pytorch3d.Transform3d, camera2liver in pytorch space. (N, 4, 4)
    :param p2c: pytorch3d.Transform3d, probe2camera in pytorch space. (N, 4, 4)
    :param verts_probe: torch.Tensor, (N, B, 3), batch of probe vertices.
    :param batch: int, (N), size of batch.
    :return: verts_probe_unbatched, List[torch.Tensor], Nx (1, B, 3)
    """
    verts_probe_transformed = p2c.compose(c2l).transform_points(verts_probe) # (N, P_probe, 3)
    verts_liver_transformed = c2l.transform_points(verts_liver) # (N, P_liver, 3)
    return verts_liver_transformed, verts_probe_transformed

def FRE_between_verts(verts_gt, verts_pred):
    """
    Calculate FRE between two lists of vertices
    """
    dist_squared = torch.square(verts_gt - verts_pred) # B, N, 3
    dist_squared_sum = torch.sum(dist_squared, axis=-1) # B, N
    return dist_squared_sum

def calculate_FRE_between_verts(c2l_gt, p2c_gt, c2l_pred, p2c_pred, verts_probe, verts_liver):
    """
    Returns the FRE between two meshes in a scene
    """
    verts_liver_gt, verts_probe_gt = verts_transformed_from_preds(c2l_gt, p2c_gt, verts_probe, verts_liver)
    verts_liver_pred, verts_probe_pred = verts_transformed_from_preds(c2l_pred, p2c_pred, verts_probe, verts_liver)
    FRE_liver_vector = FRE_between_verts(verts_liver_gt, verts_liver_pred) # B, N
    FRE_probe_vector = FRE_between_verts(verts_probe_gt, verts_probe_pred) # B, N
    return FRE_liver_vector, FRE_probe_vector
