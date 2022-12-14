# coding=utf-8

"""
Test module for generators/test_augmentation.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from lapvideous_pt.generators.augmentation.image_space_aug import delete_feature, delete_batch_feature, delete_channel_features


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
    feature_del_3 = delete_batch_feature(test_tensor, 100, 1, 3, 3, "cpu")
    assert np.array_equal(np.squeeze(feature_del_3.numpy()),
                    np.squeeze(test_tensor_no_3.numpy()))

def test_detect_delete_components_channel():
    """
    Test to check the correct items are deleted from
    a test image tensor with different sized features.
    """
    test_tensor = torch.from_numpy(np.array(
                    [
                      [[[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],
                      
                      [[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],
                      
                      [[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],
                      
                      ],

                      [[[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 1, 1]],

                      [[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 1, 1]],

                      [[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 1, 1]]]
                    ]
    )).float() # 2, 3, 4, 7

    test_tensor_no_3 = torch.from_numpy(np.array(
                    [
                      [[[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],
                      
                      [[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],
                      
                      [[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],
                      ],

                      [[[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],

                      [[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 1, 1]],

                      [[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]]
                    ]
    )).float() # 2, 3, 4, 7

    test_tensor_no_3_1 = torch.from_numpy(np.array(
                    [
                      [[[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],
                      
                      [[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],
                      
                      [[1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]],
                      ],

                      [[[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]],

                      [[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 1, 1]],

                      [[1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 1]]]
                    ]
    )).float() # 2, 3, 4, 7
    feature_del_3_1 = delete_channel_features(test_tensor,
                                            [True, False, True],
                                            100,
                                            [1, None, 1],
                                            [3, None, 1],
                                            [3, None, 1],
                                            "cpu")
    feature_del_3 = delete_channel_features(test_tensor,
                                            [True, False, True],
                                            100,
                                            [1, None, 1],
                                            [3, None, 3],
                                            [3, None, 3],
                                            "cpu")
    assert np.array_equal(np.squeeze(feature_del_3_1.numpy()),
                    np.squeeze(test_tensor_no_3_1.numpy()))
    assert np.array_equal(np.squeeze(feature_del_3.numpy()),
                    np.squeeze(test_tensor_no_3.numpy()))
