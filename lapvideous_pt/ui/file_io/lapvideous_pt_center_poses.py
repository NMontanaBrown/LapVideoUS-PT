# coding=utf-8

"""
CLI to convert a folder of gt poses into
mean centered poses.
"""

import os
import argparse
import numpy as np
import lapvideous_pt.generators.utils as rsu

def center_folder(folder_path,
                  folder_save,
                  liver_offset,
                  susi_2_p2l,
                  gl_to_cv):
    """
    Takes a folder full of GT registrations in the
    form of 4x4 poses, converts them to centered
    space.

    :param folder_path: str, path where GT poses are stored.
    :param folder_save: str, path to save.
    :param liver_offset: str, path to spp_liver_offset.txt file.
    :param susi_2_p2l: bool, whether or not to convert from US ->
                       SmartLiver representation
    :param gl_to_cv: bool whether or not to convert from OpenGL ->
                     OpenCV coordinate representation.
    """
    poses_gt_files = [item for item in os.listdir(folder_path) if ".txt" in item]
    path_poses = [os.path.join(folder_path, item) for item in poses_gt_files]
    offset = np.loadtxt(liver_offset)
    if folder_save:
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)

    for i, pose in enumerate(path_poses):
        # Generate mean centered pose
        pose_name = poses_gt_files[i].split("_")
        if susi_2_p2l:
            pose_centered = np.loadtxt(pose,delimiter=",")
            # Convert to p2l for SmartLiver
            pose_centered = rsu.slicesampler_gt_2_p2l(pose_centered, offset)
            pose_new_name = "_".join([pose_name[0], "centered", pose_name[1]])
        elif gl_to_cv:
            # Converting OpenGL matrices to OpenCV matrices.
            pose_centered = np.loadtxt(pose)
            # l2c in OpenCV
            pose_centered = rsu.opencv_to_opengl_extrinsics(pose_centered)
            # Apply offset to c2l.
            pose_centered = rsu.non_norm_2_normalised_space(np.linalg.inv(pose_centered), offset)
            pose_centered = np.linalg.inv(pose_centered) # Back to l2c
            pose_new_name = "_".join([pose_name[0], "centered", pose_name[1], pose_name[-1]])
        else:
            # Just offset into mean-centered liver space.
            pose_centered = np.loadtxt(pose)
            pose_centered = rsu.non_norm_2_normalised_space(pose_centered, offset)
            pose_new_name = "_".join([pose_name[0], "centered", pose_name[1], pose_name[-1]])

        # Generate save path
        path_save = os.path.join(folder_path, pose_new_name)
        # Save file
        if folder_save:
            path_save = os.path.join(folder_save, pose_new_name)
        np.savetxt(path_save, pose_centered)

if __name__ == "__main__":
    """
    Entry point for LapVideoUS center poses
    """

    parser = argparse.ArgumentParser(description='LapVideoUS_center_poses')

    ## ARGS
    parser.add_argument("--path_files",
                        "-pf",
                        type=str,
                        required=True,
                        help="Path where all pose files")

    parser.add_argument("--path_save",
                        "-ps",
                        type=str,
                        required=False,
                        default=None,
                        help="Path save")

    parser.add_argument("--mean_center_file",
                        "-l",
                        required=False,
                        default=None,
                        help="path to spp_liver_offset.txt")

    parser.add_argument("--susi_2_p2l",
                        "-p2l",
                        type=bool,
                        required=False,
                        default=False,
                        help="susi_2_p2l switch")

    parser.add_argument("--gl_to_cv",
                        "-gl",
                        type=bool,
                        required=False,
                        default=False,
                        help='opengl_switch')

    args = parser.parse_args()
    center_folder(args.path_files,
                  args.path_save,
                  args.mean_center_file,
                  args.susi_2_p2l,
                  args.gl_to_cv)