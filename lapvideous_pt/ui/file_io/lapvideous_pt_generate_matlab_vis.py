# coding=utf-8

"""
Generating Data For MATLAB Visualisations
"""

import os
import scipy.io as sio
import json
import torch
import vtk
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter 
import matplotlib.pyplot as plt
import lapvideous_pt.generators.video_generation.utils as vru
from lapvideous_pt.models.models import LapVideoUS
import lapvideous_pt.io.matlab_visualisation as mv
from pytorch3d.transforms import Transform3d
import slicesampler.pycuda_simulation.segmented_volume as svol
import argparse

def switch_hepatic_portal_arteries(image_tensor):
    """
    Function to change the colours of the
    vessel features
    :param image_tensor: np.Array, [B, W, H, Ch]
    :return: np.Array, [B, W, H, Ch]
    """
    # Features in image_tensor are ordered:
    hv = image_tensor[:, :, :, 0]
    pv = image_tensor[:, :, :, 1]
    a = image_tensor[:, :, :, 2]
    new_image = np.zeros_like(image_tensor)
    new_image[:, :, :, 1] = hv # Make HV blue
    new_image[:, :, :, 2] = pv # Make PV green
    new_image[:, :, :, 0] = a # Make arteries Y
    new_image[:, :, :, 1] += a
    return new_image


def get_and_save_vis(model,
                     transform_l2c,
                     transform_p2c,
                     liver_data,
                     lus_g,
                     batch,
                     index,
                     path_save_vis,
                     name_file,
                     load_model=None,
                     name_pred=None):
    """
    Function to generate the data for visualisation.
    :param model: LapVideoUS object, model containing trained weights and rendering
                  functions.
    :param transform_l2c: Transform3d, batch of transforms to render
    :param transform_p2c: Transform3d, batch of transforms to render
    :param liver_data: List[torch.Tensor]
    :param lus_g: slicesampler object
    :param batch: int
    :param index: int
    :param path_save_vis: str, path to save
    :param name_file: List[str], name of file to save
    :param load_model: str or None, path of weights to save.
    :param name_pred: List[str] or None, name of files to save
    """
    l2c_cv, p2l_cv = vru.get_cv_matrices(l2c=transform_l2c, p2c=transform_p2c, device=model.device)

    # Generate images from the model.
    image_tensor, _ = model.render_data(liver_data, transform_l2c.get_matrix(), transform_p2c.get_matrix())

    if load_model:
        outputs = model.forward(liver_data, (image_tensor).to(model.device))
        tensor_pred = outputs[0]
        p2l = outputs[1][1]
        c2l = outputs[1][0]
        l2c_cv_p, p2l_cv_p = vru.get_cv_matrices(c2l.inverse(), p2l=p2l, device=model.device)
    
    # Lists to save data.
    p2l_slicesampler = vru.batch_p2l_2_slicesampler(p2l_cv.detach().cpu().numpy())
    if load_model:
        p2l_slicesampler_p = vru.batch_p2l_2_slicesampler(p2l_cv_p.detach().cpu().numpy())

    image_tensor = np.transpose(image_tensor.detach().cpu().numpy(), [0, 3, 2, 1])
    us = image_tensor[:, :, :, 3:]
    us_processed = switch_hepatic_portal_arteries(us)
    final_gt_vis = np.concatenate((image_tensor[:,:,:,0:3], us_processed), axis=-1)
    tensor_pred = np.transpose(tensor_pred.detach().cpu().numpy(), [0, 3, 2, 1])

    render_items = mv.sim_slicesampler_plane(batch, lus_g, p2l_slicesampler, l2c_cv, p2l_cv)
    mv.save_formatted_data(path_save_vis, name_file[index], final_gt_vis, render_items, lus_g)
    if load_model:
        us = tensor_pred[:, :, :, 3:]
        us_processed = switch_hepatic_portal_arteries(us)
        final_pred_vis = np.concatenate((tensor_pred[:,:,:,0:3], us_processed), axis=-1)
        preds = mv.sim_slicesampler_plane(batch, lus_g, p2l_slicesampler_p, l2c_cv_p, p2l_cv_p)
        mv.save_formatted_data(path_save_vis, name_pred[index], final_pred_vis, preds, lus_g, names=["ims_pred", "planes_pred", "cam_pose_pred", "lus_pose_pred"])
    

def generate_matlab_vis(path_lus_json,
                        path_lus_files,
                        path_save_vis,
                        expt_config,
                        batch,
                        load_model=None):
    """
    Function to generate a series of files
    for stop motion animation of network prediction
    for each degree of freedom.
    :param path_lus_json: path to model_lus.json 
    :param path_lus_files: path to VTK files for slicesampler
    :param path_save_vis: path to save for simulation objects
    :param expt_config: path to model config.
    :param batch: int, batch size for simulation
    :param load_model: str | None, path to trained model to test at
                       inference.
    """
    # paths to relevant stuff
    model_lus_json = os.path.join(path_lus_json)

    # Folder where VTK files are
    model_lus_sim = os.path.abspath(path_lus_files)

    # Instantiate a generator via a model
    model_pt_expt_config = expt_config
    with open(model_pt_expt_config) as f:
        expt_config = json.load(f)
    model = LapVideoUS(**expt_config)
    if load_model:
        model.load_state_dict(torch.load(load_model))
    model.batch = batch
    liver_data = model.prep_input_data_for_render()

    transform_l2c = model.video_loader.l2c.expand(model.batch, -1, -1) # (N, 4, 4)
    transform_p2c = model.video_loader.p2c.expand(model.batch, -1, -1) # (N, 4, 4)

    # Generate some poses
    names_p2l = ['final1_mat_rx_p2l.mat','final1_mat_ry_p2l.mat','final1_mat_rz_p2l.mat','final1_mat_tx_p2l.mat','final1_mat_ty_p2l.mat','final1_mat_tz_p2l.mat',]
    if load_model:
        names_p2l_pred = ["pred_"+ item for item in names_p2l]
    names_l2c = ['final1_mat_rx_l2c.mat','final1_mat_ry_l2c.mat','final1_mat_rz_l2c.mat','final1_mat_tx_l2c.mat','final1_mat_ty_l2c.mat','final1_mat_tz_l2c.mat',]
    if load_model:
        names_l2c_pred = ["pred_"+ item for item in names_l2c]

    # Generate data - using slicesampler as we can get the point planes like this.
    lus_g = svol.SegmentedVolume(config_dir=model_lus_json,
                                    mesh_dir=model_lus_sim,
                                    voxel_size=0.5,
                                    downsampling=2,
                                    image_num=1)
    # Simulate moving p2l
    for j in range(6):
        batch_perturbations = vru.generate_ordered_params_index(model.device, batch, [j], [-10], [10])
        transforms_r_p2l, transforms_t_p2l = vru.generate_transforms(model.device, model.batch, batch_perturbations)
        transforms_r_c2l = Transform3d(device=model.device) # Identity matrix
        transforms_t_c2l = Transform3d(device=model.device) # Identity matrix
        transform_l2c_perturbed, transform_p2c_perturbed = vru.perturb_orig_matrices_in_CV_space(Transform3d(matrix=transform_l2c.clone(), device=model.device),
                                                                                                Transform3d(matrix=transform_p2c.clone(), device=model.device),
                                                                                                transforms_r_c2l,
                                                                                                transforms_t_c2l,
                                                                                                transforms_r_p2l,
                                                                                                transforms_t_p2l,
                                                                                                model.device)
        get_and_save_vis(model, transform_l2c_perturbed, transform_p2c_perturbed, liver_data, lus_g, batch, j, path_save_vis, names_l2c, load_model, names_l2c_pred)

    # Simulate moving l2c
    for j in range(6):
        batch_perturbations = vru.generate_ordered_params_index(model.device, batch, [j], [-10], [10])
        transforms_r_c2l, transforms_t_c2l = vru.generate_transforms(model.device, model.batch, batch_perturbations)
        transforms_r_p2l = Transform3d(device=model.device) # Identity matrix
        transforms_t_p2l = Transform3d(device=model.device) # Identity matrix
        transform_l2c_perturbed, transform_p2c_perturbed = vru.perturb_orig_matrices_in_CV_space(Transform3d(matrix=transform_l2c.clone(), device=model.device),
                                                                                                Transform3d(matrix=transform_p2c.clone(), device=model.device),
                                                                                                transforms_r_c2l,
                                                                                                transforms_t_c2l,
                                                                                                transforms_r_p2l,
                                                                                                transforms_t_p2l,
                                                                                                model.device)
        get_and_save_vis(model, transform_l2c_perturbed, transform_p2c_perturbed, liver_data, lus_g, batch, j, path_save_vis, names_p2l, load_model, names_p2l_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MATLAB Visualisation files.')
    parser.add_argument("--path_lus_json",
                        "-d",
                        help="Path to model_lus.json config",
                        required=True,
                        type=str,
                        default=None)
    parser.add_argument("--meshdir",
                        "-md",
                        help="Path to lus simulation data",
                        required=True,
                        type=str,
                        default=None)
    parser.add_argument("--path_save",
                        "-ps",
                        help="Path to save",
                        required=True,
                        type=str,
                        default=None)
    parser.add_argument("--config_path",
                        "-conf",
                        help="Which configuration file to instantiate model",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("--load_model",
                        "-lm",
                        help="Which state-dict path to use to predict",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("--batch",
                        "-b",
                        help="Batch size for simulation",
                        required=False,
                        type=int,
                        default=10)

    args = parser.parse_args()

    generate_matlab_vis(path_lus_json=args.path_lus_json,
                        path_lus_files=args.meshdir,
                        path_save_vis=args.path_save,
                        batch=args.batch,
                        expt_config=args.config_path,
                        load_model=args.load_model)