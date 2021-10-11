# coding=utf-8

"""
Command line to pre-process the ultrasound volumes to
np arrays prior to training.
"""

import os
import json
import numpy as np
import argparse
from lapvideous_pt.generators.ultrasound_reslicing.us_file_io import USTensorSlice

def prep_US_data(config_file:str,
                 pose:str=None):
    """
    Function to prepare the US volumes for training
    of networks.
    :param config_file: str, path to config file for
                        US generation.
    :param pose: str, default=None, path to US gt p2l
                 slicesampler format to render against
                 from slicesampler and PT rendering.
    """
    with open(config_file) as f:
        data = json.load(f)

    file_io = USTensorSlice(**data)
    if pose:
        pose = np.loadtxt(pose, delimiter=',')
        file_io.slice_and_compare(pose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LapVideoUS_pt_US_preprocessing')

    ## ARGS
    parser.add_argument("--path_json_config",
                        "-pj",
                        required=True,
                        type=str,
                        help="Path to config .json for simulation")
    parser.add_argument("--pose",
                        "-p",
                        required=False,
                        type=str,
                        default=None,
                        help="Path to pose file to compare results.")

    args = parser.parse_args()

    prep_US_data(os.path.abspath(args.path_json_config),
                 os.path.abspath(args.pose))
