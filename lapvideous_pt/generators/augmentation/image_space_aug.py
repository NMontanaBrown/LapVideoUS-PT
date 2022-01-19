# coding=utf-8

"""
Module to implement generation of image space augmentations,
such as feature addition/deletion, erosion, dilation.
"""

import os
import torch
import numpy as np
from kornia.contrib import connected_components as cc

def delete_feature(image, num_iterations, num_features_del, min_size_features, max_size_features, device):
    """
    Given an image tensor, find all connected components,
    then, pick a number of them and erase them from the
    image.
    :param image: torch.Tensor
    :param num_iterations: int, 
    :param num_features_del: int,
    :param min_size_features: int, minimum area (pixels) of image that can be deleted
    :param max_size_features: int, max area (pixels) of image that can be deleted
    :param device:
    :return: torch.Tensor
    """
    # Detect objects in the image
    connected_components = cc(image, num_iterations)
    output = torch.unique(connected_components,
                             sorted=True) # torch.Tensor

    # Get individual features as masks.
    individual_features = [torch.eq(connected_components, item) for item in torch.split(output, 1)]
    
    # Get areas of features
    area_features = [torch.sum(item).expand(1) for item in individual_features]
    area_tensor = torch.cat(area_features)
    area_features_above_min = torch.where(area_tensor>=min_size_features, torch.ones(1).to(device).float(), torch.zeros(1).to(device).float())
    area_features_below_max = torch.where(area_tensor<=max_size_features, torch.ones(1).to(device).float(), torch.zeros(1).to(device).float())
    area_features_thresh = torch.multiply(area_features_above_min, area_features_below_max) # satisfies both conditions
    area_features_thresh[0] = torch.zeros(1).to(device).float() # The first, as it is always 0 - the background, but we do not want to choose it.

    # Select features
    # depending on num_features_del and num individual connected components
    # detected in image that satisfy area constraints.
    choice_indices = torch.nonzero(area_features_thresh)
    # Check how many choices there are, if lower than num_features_del, use all.
    if num_features_del > output.shape[0]:
        indices = output.shape
    else:
        indices = num_features_del

    chosen_indices = torch.randperm(choice_indices.shape[0])[:indices]
    final_indices = torch.zeros(output.shape).to(device)
    final_indices[choice_indices[chosen_indices]] = torch.ones(1).to(device)
    choices = torch.split(final_indices, 1)

    # Modify individual features
    modified_features = [torch.zeros_like(feature).to(device) if choices[i]==torch.ones(1).to(device) else feature for i, feature in enumerate(individual_features)]
    final_image = torch.sum(torch.cat(modified_features[1:]), 0) # Don't sum the background
    return final_image

def delete_batch_feature(image, num_iterations, num_features_del, min_size_features, max_size_features, device):
    """
    Given an image tensor, find all connected components,
    then, pick a number of them and erase them from the
    image.
    :param image: torch.Tensor
    :param num_iterations: int, 
    :param num_features_del: int,
    :param min_size_features: int, minimum area (pixels) of image that can be deleted
    :param max_size_features: int, max area (pixels) of image that can be deleted
    :param device:
    :return: torch.Tensor
    """
    # Split batchwise
    batch_split_image = torch.split(image, 1)
    # Modify each batch separately
    modified_ims = [delete_feature(item,
                                   num_iterations,
                                   num_features_del,
                                   min_size_features,
                                   max_size_features,
                                   device) for item in batch_split_image]
    batch_join = torch.cat(modified_ims, dim=0) # Join batchwise
    return batch_join

def add_feature(image, num_iterations, num_features_add, min_size_features, max_size_features):
    """
    Given an image tensor, add random
    features to the image.
    :param image:
    :param num_iterations:
    :param num_features_del:
    :param min_size_features:
    :param max_size_features:
    """
    return None

def generate_kernel(kernel_value, range_value:int, proba:float, device):
    """
    Given probability values proba, if
    random probability is higher than proba, generate a randomly
    sized ones tensor in range kernel_value -> range_value 
    for a gaussian morphological operation on device.

    :param kernel_value: int or None
    :param range_value: int
    :param proba: float [0-1]
    :param device: str, cuda or cpu
    :return: kernel, torch.Tensor
    """

    if kernel_value is not None:
        if np.random.rand() > proba:
            interval = np.random.choice(np.arange(kernel_value, kernel_value+range_value, 1))
            kernel = torch.ones(interval, interval).to(device)
        else:
            kernel = None
    else:
        kernel = None
    return kernel

def generate_erosion_dilation_kernels_lists(us_erosion_dil_list,
                                            vid_erosion_dil_list,
                                            proba_us,
                                            proba_vid,
                                            device):
    """
    Generate all the parameters for test or train time augmentation in the image
    space on us or video images.

    :param us_erosion_dil_list: List[int | None], (2,), erosion, dilation for US images.
                                None will be interpreted
                                as no operation to be returned.
    :param vid_erosion_dil_list: List[int | None], (2,), erosion, dilation for vid images.
                                 None will be interpreted
                                 as no operation to be returned.
    :param proba_us: float, [0-1], representing probability above which kernel will be
                     generated for us.
    :param proba_vid: float, [0-1], representing probability above which kernel will be
                     generated for vid.
    :param device: str, cuda or cpu onto which one should place the devices.

    :return: []
    """
    # US - generate erosion
    us_erosion_kernel = generate_kernel(us_erosion_dil_list[0], 1, proba_us, device)
    us_dil_kernel = generate_kernel(us_erosion_dil_list[1], 1, proba_us, device)
    us_kernels = [us_erosion_kernel, us_dil_kernel, None]

    vid_erosion_kernel = generate_kernel(vid_erosion_dil_list[0], 2, proba_vid, device)
    vid_dil_kernel = generate_kernel(vid_erosion_dil_list[1], 2, proba_vid, device)
    vid_kernels = [vid_erosion_kernel, vid_dil_kernel, None]

    # Return kernels for augmentation.
    return us_kernels, vid_kernels
