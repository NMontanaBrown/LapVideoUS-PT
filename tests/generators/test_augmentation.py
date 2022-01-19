# coding=utf-8

"""
Test module for generators/test_augmentation.py
"""

import pytest
import torch
import numpy as np
from kornia.contrib import connected_components as cc
import matplotlib.pyplot as plt
from lapvideous_pt.generators.augmentation.image_space_aug import delete_feature, delete_batch_feature


def test_detect_delete_components():
    """
    Test to check the correct items are deleted from
    a test image tensor with different sized features.
    """
    test_tensor = torch.from_numpy(np.array(
                    [[[[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]]]
    )).float()
    test_tensor_no_3 = torch.from_numpy(np.array(
                    [[[[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]]]
    )).float()
    test_tensor_no_1 = torch.from_numpy(np.array(
                    [[[[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]]]
    )).float()
    test_tensor_no_2 = torch.from_numpy(np.array(
                    [[[[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]]]
    )).float()
    feature_del_3 = delete_feature(test_tensor, 100, 1, 3, 3, "cpu")
    feature_del_2 = delete_feature(test_tensor, 100, 1, 2, 2, "cpu")
    feature_del_1 = delete_feature(test_tensor, 100, 1, 1, 1, "cpu")

    assert np.array_equal(np.squeeze(feature_del_3.numpy()),
                    np.squeeze(test_tensor_no_3.numpy()))

    assert np.array_equal(np.squeeze(feature_del_2.numpy()),
                    np.squeeze(test_tensor_no_2.numpy()))

    assert np.array_equal(np.squeeze(feature_del_1.numpy()),
                    np.squeeze(test_tensor_no_1.numpy()))

def test_detect_delete_components_batch():
    """
    Test to check the correct items are deleted from
    a test image tensor with different sized features.
    """
    test_tensor = torch.from_numpy(np.array(
                    [[[[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]],
                      [[[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 1, 1]]]
                    ]
    )).float() # 2,1, 4, 7
    test_tensor_no_3 = torch.from_numpy(np.array(
                    [[[[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]],
                      [[[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]]
                    ]
    )).float()
    print("expected shape", test_tensor_no_3.shape)
    feature_del_3 = delete_batch_feature(test_tensor, 100, 1, 3, 3, "cpu")
    print("test expect result", test_tensor_no_3)
    print("cc result", feature_del_3)
    assert np.array_equal(np.squeeze(feature_del_3.numpy()),
                    np.squeeze(test_tensor_no_3.numpy()))
    