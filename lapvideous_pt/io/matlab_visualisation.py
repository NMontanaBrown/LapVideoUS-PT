# coding=utf-8

"""
Functions to generate visualisations for MATLAB
"""

import os
import torch
import argparse
import numpy as np
import scipy.io as sio
import slicesampler.pycuda_simulation.segmented_volume as svol
import lapvideous_pt.io.utils as rio
import lapvideous_pt.generators.video_generation.utils as vru

def generate_matlab_vis_files(path_to_files,
                              save_folder):
    """
    Function to generate visualisation from a folder
    with files.

    It uses the generated liver2camera poses and probe2camera
    poses from the SmartLiver app to generate a Slicesampler
    plane and check that it is sensible. Displays the
    model in 3D and the LUS plane segmentations.

    :param path_to_files: str, path where all the generation
                          data is stored.
    :param save_folder: str, path where the .mat files will
                        will be saved
    """
    # Path to needed files, generated from folder
    path_to_mesh = os.path.join(path_to_files, "VTK")
    path_to_json = os.path.join(path_to_files, "models_lus.json")
    # The p2l pose, in susi format. Should be centered,
    # as we are using the liver_normalised mesh.
    l2c = np.loadtxt(os.path.join(path_to_files, "spp_liver2camera.txt"))
    p2c = np.loadtxt(os.path.join(path_to_files, "spp_probe2camera.txt"))
    p2l = np.linalg.inv(l2c) @ p2c
    p2l_slicesampler = vru.p2l_2_slicesampler_numpy(p2l)

    # Which surface model to generate
    list_surfaces = ["spp_liver_normalised",
                     "arteries",
                     "hepatic_veins",
                     "portal_vein"]
    list_save_names = ["liver.mat",
                       "arteries.mat",
                       "hepatic.mat",
                       "portal.mat"]
    list_var_names = ["spp_liver_normalised",
                     "arteries",
                     "hepatic",
                     "portal"]

    # Path saves
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    path_save_vertices = save_folder

    path_save_plane = save_folder
    plane_name_save = 'plane.mat'
    plane_var = 'plane'

    image_name_save = "lus.mat"
    image_var = "lus"

    path_save_quiver = save_folder
    quiver_name = 'vector.mat'
    quiver_var = 'vectors'

    # Simulate your mesh
    liver_volume = svol.SegmentedVolume(config_dir=path_to_json,
                                        mesh_dir=path_to_mesh,
                                        voxel_size=0.5,
                                        downsampling=2,
                                        image_num=1)

    # Get vertices and save
    for i, item in enumerate(list_surfaces):
        surface_vertices = liver_volume.meshes[item].vertices
        rio.make_mat_file_mesh(surface_vertices,
                               path_save_vertices,
                               list_save_names[i],
                               list_var_names[i])

    # Simulate plane and save
    points, binary_map, colored_map = \
        liver_volume.simulate_image(poses=p2l_slicesampler, image_num=1, out_points=True)
    rio.make_mat_files_plane(colored_map, points, path_save_plane, plane_name_save, plane_var)
    rio.make_mat_file_mesh(binary_map,
                           path_save_vertices,
                           image_name_save,
                           image_var)

    # Get quiver3 points and save
    quiver_save = rio.pose_2_quiver(p2l_slicesampler)
    rio.make_mat_file_mesh(quiver_save, path_save_quiver, quiver_name, quiver_var)


def sim_slicesampler_plane(batch,
                           lus_g,
                           p2l_slicesampler,
                           poses_l2c_cv,
                           poses_p2l_cv):
    """
    Function to generate MATLAB friendly arrays for
    visualisation of registration.
    :param batch: int
    :param lus_g: slicesamplre object
    :param p2l_slicesampler: np.array
    :param poses_l2c_cv: torch.Tensor,
    :param poses_p2l_cv: torch.Tensor,
    :return: - points_save:
             - binary_map_save:
             - colored_map_save:
             - cam_pose:
             - lus_pose:
             - quiver_lus:
             - quiver_cam:
    """
    points_save = []
    binary_map_save = []
    colored_map_save = []
    cam_pose = []
    lus_pose = []
    quiver_lus = []
    quiver_cam = []
    for i in range(batch):
        # # Use the slicesampler object to simulate objects in 3D
        points, binary_map, colored_map =\
            lus_g.simulate_image(poses=p2l_slicesampler[:,i*4:(i+1)*4],
                                image_num=1,
                                out_points=True)

        points_per_plane = int(points.shape[0]/colored_map.shape[3])
        plane_save = points[:points_per_plane:10, :]
        points_save.append(plane_save)
        binary_map_save.append(binary_map)
        colored_map_save.append(colored_map)
        # Gen positions of items
        cam_pose.append(np.linalg.inv(poses_l2c_cv.detach().cpu().numpy()[i, :, :])[0:3, 3])
        lus_pose.append(poses_p2l_cv.detach().cpu().numpy()[i, 0:3, 3])
        # Generate 3D quiver arrows
        quiver_lus.append(rio.pose_2_quiver(poses_p2l_cv.detach().cpu().numpy()[i, :, :]))
        quiver_cam.append(rio.pose_2_quiver(poses_l2c_cv.detach().cpu().numpy()[i, :, :]))

    return points_save, binary_map_save, colored_map_save, cam_pose, \
        lus_pose, quiver_lus, quiver_cam

def save_formatted_data(path_save_vis,
                        filename,
                        image_tensor,
                        render_items,
                        lus_g,
                        names=None):
    """
    Function to save data in standard format to MATLAB
    rendering. 
    :param path_save_vis: str, path to save
    :param filename: str, name of file.
    :param image_tensor: torch.Tensor, [B, CH, W, H] of image data
    :param render_items: List[list], output of sim_slicesampler_plane
    :param lus_g: slicesampler object.
    :param names: List[str] or None, defining variable names
                  of rendered images, us planes, cam pose and us pose.
    """
    if names is None:
        names = ["ims", "planes", "cam_pose", "lus_pose"]

    sio.savemat(os.path.join(path_save_vis, filename),
                    {names[0]: image_tensor,
                    names[1]: render_items[0],
                    "liver_points": lus_g.meshes['spp_liver_normalised'].vertices,
                    "HV_points": lus_g.meshes['hepatic_veins'].vertices,
                    "PV_points": lus_g.meshes['portal_vein'].vertices,
                    "arteries_points": lus_g.meshes['arteries'].vertices,
                    "liver_faces": lus_g.meshes['spp_liver_normalised'].faces,
                    "HV_faces": lus_g.meshes['hepatic_veins'].faces,
                    "PV_faces": lus_g.meshes['portal_vein'].faces,
                    "arteries_faces": lus_g.meshes['arteries'].faces,
                    names[2]: render_items[3],
                    names[3]: render_items[4],
                    "quiver_lus":render_items[5],
                    "quiver_cam" : render_items[6]
                    }
                    )
