# coding=utf-8

"""
CLI to pre-generate p2l poses over the surface of a
given object surface given a fixed l2c reference,
and a sample p2l reference.
"""

import os
import argparse
import numpy as np
import lapvideous_pt.generators.video_generation.utils as vru
import lapvideous_pt.generators.ultrasound_reslicing.utils as guru

def generate_p2c_files(path_to_l2c:str,
                       path_to_p2l:str,
                       path_vtk:str,
                       num_poses:int,
                       path_save:str):
    """
    Function to generate p2c files
    for fixed l2c and a reference p2l file
    in proximity to the location of the
    p2l file.

    :param path_to_l2c: str, path to l2c reference pose.
    :param path_to_p2l: str, path to p2l reference pose.
    :param path_vtk: str, path to .vtk model file.
    :param num_poses: int, number of poses to pre-generate
    :param path_save: str, path to save.
    """
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    p2l = np.loadtxt(path_to_p2l)
    l2c = np.loadtxt(path_to_l2c)
    p2l_slice = vru.p2l_2_slicesampler_numpy(p2l)

    vertices, _, normals = guru.get_model_vertices_faces_normals(path_vtk)
    closest_poses, distance = guru.generate_close_poses(p2l_slice, vertices, normals, num_poses)
    print("Furthest sample (not inclusive) mm", distance[num_poses])

    # Save p2c for rendering / training.
    for i in range(num_poses):
        p2c_pose_path = os.path.join(path_save, "p2c_{}_auto.txt".format(i))
        p2c = l2c @  vru.slicesampler_2_p2l_numpy(closest_poses[i, :, :])
        np.savetxt(p2c_pose_path, p2c)

if __name__ == "__main__":
    """
    Entry point for LapVideoUS generate p2c poses for training / rendering.
    """

    parser = argparse.ArgumentParser(description='LapVideoUS_generate_p2c_poses')

    ## ARGS
    parser.add_argument("--path_l2c",
                        "-pl2c",
                        type=str,
                        required=True,
                        help="Path for l2c for simulation")

    parser.add_argument("--path_p2l",
                        "-pp2l",
                        type=str,
                        required=True,
                        help="Path for p2l for simulation")

    parser.add_argument("--path_save",
                        "-ps",
                        type=str,
                        required=False,
                        default=None,
                        help="Path save")

    parser.add_argument("--path_vtk",
                        "-vtk",
                        type=str,
                        required=False,
                        default=None,
                        help="Path model")

    parser.add_argument("--num_poses",
                        "-n",
                        type=int,
                        required=False,
                        default=10,
                        help="Number of poses to pre-generate")


    args = parser.parse_args()
    generate_p2c_files(args.path_files,
                       args.path_save,
                       args.mean_center_file,
                       args.susi_2_p2l,
                       args.gl_to_cv)
