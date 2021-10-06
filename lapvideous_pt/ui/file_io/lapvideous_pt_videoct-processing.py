# coding=utf-8

"""
CLI to convert the Video-CT Ground truth OpenGL txt files
and poses into extrinsic matrices for simulation purposes.
"""

import os
import argparse
import numpy as np

def process_video_ct_file(path_file):
    """
    Function to process a single .txt file
    into a 4x4 np array of extrinsic l2c parameters.

    :param path_file: .txt file, containing OpenGL parameters.
    :return: [np.array, (4, 4), str filename]
    """
    if not os.path.isfile(path_file):
        raise ValueError("{} is not a file".format(path_file))
    
    values = np.zeros((4,4))
    # Read file
    for index, line in enumerate(reversed(list(open(path_file)))):
        # Only read last 4 lines.
        if index == 4:
            break
        val_str = line.split(",")
        vals = [float(val) for val in val_str]
        values[index,:] = vals
        
    # Reverse axis, as we have filled it in from the top.
    values = np.flipud(values)
    return values

def generate_filename_videoct(path_file):
    """
    Generates a filename from input txt
    string
    :param path_file: str, path to file.
    """
    split_list = path_file.split('/')
    split_name = split_list[-1].split("_")
    name = "_".join(["image", split_name[1], split_name[2], split_name[-1]])

    return name

def lapvideous_video_ct_folder_conversion(folder_matrices,
                                          path_save=None):
    """
    :param folder_matrices: str, path to where all the video ct files are stored.
    :param path_save: where the new files will be stored.
    """
    if not os.path.exists(folder_matrices):
        raise ValueError
    files = os.listdir(folder_matrices)
    # Only get ".txt" files
    path_files = [os.path.join(folder_matrices, file) for file in files if ".txt" in file]
    for item in path_files:
        extrinsics = process_video_ct_file(item)
        name = generate_filename_videoct(item)
        # Save file
        if path_save:
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            final_path = os.path.join(path_save, name)
        else:
            final_path = os.path.join(folder_matrices, name) 
        np.savetxt(final_path, extrinsics)

if __name__ == "__main__":
    """
    Entry point for LapVideoUS Video CT processing
    """

    parser = argparse.ArgumentParser(description='LapVideoUS_process_videoct_opengl')

    ## ARGS
    parser.add_argument("--path_files",
                        "-pf",
                        type=str,
                        required=True,
                        help="Path where all opengl files saved.")

    parser.add_argument("--path_save",
                        "-ps",
                        required=False,
                        default=None,
                        help="Where to store files")


    args = parser.parse_args()
    lapvideous_video_ct_folder_conversion(args.path_files,
                                          args.mean_center_file,
                                          args.path_save)