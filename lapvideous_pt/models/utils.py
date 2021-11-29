# coding=utf-8

"""
Module containing utils to parse model architecture.
"""

import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList

def parse_model_config_convs(model_config_dict):
    """
    Creates torch nn.ModuleLists from model configuration
    dictionary. Specifically will parse the values in the
    dictionary for a conv backbone.
    :param model_config_dict: dict,
    :return: [torch.nn.ModuleList, List[str]]
    """
    layer_list = []
    layer_names = []
    for key in model_config_dict:
        if model_config_dict[key]["layer_type"] == "Conv2d":
            layer_list.append(nn.Conv2d(**model_config_dict[key]["params"]))
            layer_names.append("Conv2d")
        elif model_config_dict[key]["layer_type"] == "Maxpool":
            layer_list.append(nn.MaxPool2d(**model_config_dict[key]["params"]))
            layer_names.append("Maxpool")
        elif model_config_dict[key]["layer_type"] == "BatchNorm2d":
            layer_list.append(nn.BatchNorm2d(**model_config_dict[key]["params"]))
            layer_names.append("BatchNorm2d")
    return torch.nn.ModuleList(layer_list), layer_names

def parse_model_config_linear(model_config_dict):
    """
    Creates torch nn.ModuleLists from model configuration
    dictionary. Specifically will parse the values in the
    dictionary as FCN layers.
    :param model_config_dict: dict,
    :return: [torch.nn.ModuleList, List[str]]
    """
    layer_list = []
    layer_names = []
    for key in model_config_dict:
        if model_config_dict[key]["layer_type"] == "Linear":
            layer_list.append(nn.Linear(**model_config_dict[key]["params"]))
            layer_names.append("Linear")
        elif model_config_dict[key]["layer_type"] == "Dropout":
            layer_list.append(nn.Dropout(**model_config_dict[key]["params"]))
            layer_names.append("Dropout")
    return torch.nn.ModuleList(layer_list), layer_names
