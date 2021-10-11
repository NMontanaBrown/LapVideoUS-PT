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

def generate_composite_probe_and_liver_meshes(verts_liver,
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
    :param verts_liver: torch.Tensor, (B, 3), vertices of liver
                        mesh.
    :param faces_liver: torch.Tensor, (F,), faces of liver mesh. 
    :param textures_liver: 
    :param verts_probe_unbatched: List[torch.Tensor], (N,),
                                  transformed vertices in liver space.
    :param faces_probe: torch.Tensor (G,), faces of probe mesh.
    :param textures_probe:
    :param batch: int, (N), batch size.
    :return: List[Meshes], (N,)
    """
    scene_meshes = [
            join_meshes_as_scene([Meshes(verts_liver.expand(1, -1, -1), faces_liver.expand(1, -1, -1), textures_liver),
                                 Meshes(verts_probe_unbatched[i], faces_probe.expand(1, -1, -1), textures_probe)]) for i in range(batch)]
    batch_meshes = join_meshes_as_batch(scene_meshes)
    return batch_meshes
