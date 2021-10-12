# coding=utf-8

"""
Testing transformations
"""

import os
import numpy as np
import torch
from pytorch3d.transforms import Transform3d
import lapvideous_pt.generators.video_generation.utils as vru

def test_opencv_opengl_opencv():
    """
    Test transforming a reference
    opencv w2c transform and back results
    in the same matrix. 
    """
    device = torch.device("cpu")
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]

    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)

    opengl_l2c, r_gl, t_gl = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    opencv_l2c, _, _ = vru.opengl_to_opencv(r_gl, t_gl, device)
    l2c_numpy = np.squeeze(opencv_l2c.numpy())
    l2c_test = np.squeeze(test_matrix_l2c)
    assert np.allclose(l2c_numpy, l2c_test)

def test_opengl_opencv_opengl():
    """
    Test transforming a reference
    opencv w2c transform and back results
    in the same matrix. 
    """
    device = torch.device("cpu")
    # Opencv
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_p2c = np.loadtxt("tests/data/spp_probe2camera.txt")
    test_matrix_p2l = np.linalg.inv(test_matrix_l2c) @ test_matrix_p2c

    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    test_matrix_p2c = np.expand_dims(test_matrix_p2c, 0)
    test_matrix_p2l = np.expand_dims(test_matrix_p2l, 0)

    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)

    opengl_l2c, r_gl, t_gl = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    rp = test_matrix_p2c[:, :3, :3]
    tp = test_matrix_p2c[:, :3, 3]
    test_matrix_p2c_torch_r = torch.from_numpy(rp)
    test_matrix_p2c_torch_t = torch.from_numpy(tp)
    opengl_p2c, rp_gl, tp_gl = vru.opencv_to_opengl(test_matrix_p2c_torch_r, test_matrix_p2c_torch_t, device)

    rp2l = test_matrix_p2l[:, :3, :3]
    tp2l = test_matrix_p2l[:, :3, 3]

    test_matrix_p2l_torch_r = torch.from_numpy(rp2l)
    test_matrix_p2l_torch_t = torch.from_numpy(tp2l)

    opengl_p2l, rp2l_gl, tp2l_gl = vru.opencv_to_opengl_p2l(test_matrix_p2l_torch_r, test_matrix_p2l_torch_t, device)

    p2c_pytorch = Transform3d(matrix=opengl_p2c)
    l2c_pytorch = Transform3d(matrix=opengl_l2c)
    p2l_pytorch = p2c_pytorch.compose(l2c_pytorch.inverse()).get_matrix()
    p2l_opencv, _, _ = vru.opengl_to_opencv_p2l(rp2l_gl, tp2l_gl, device)
    # Check that p2l_pytorch is the same as numpy pytorch -> gl
    assert np.allclose(p2l_pytorch.numpy(), opengl_p2l.numpy())
    # Check that p2l_pytorch -> opencv is the same as og p2l from numpy
    assert np.allclose(p2l_opencv.numpy(), test_matrix_p2l)

def test_p2l_slicesampler_p2l():
    """
    Test converting from p2l -> slicesampler
    and back returns the same matrix.
    """
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_p2c = np.loadtxt("tests/data/spp_probe2camera.txt")
    test_matrix_p2l = np.linalg.inv(test_matrix_l2c) @ test_matrix_p2c
    test_matrix_p2l = np.expand_dims(test_matrix_p2l, 0)

    p2l_pytorch = torch.from_numpy(test_matrix_p2l)
    slicesampler_p2l = vru.p2l_2_slicesampler(p2l_pytorch)
    p2l_from_slicesampler = vru.slicesampler_2_p2l(slicesampler_p2l)

    assert np.allclose(p2l_from_slicesampler.numpy(), test_matrix_p2l)

def test_p2l_from_gl_to_slicesampler():
    """
    Test if p2l from l2c and p2c in pytorch
    frame can be recovered from using series
    of transforms.
    """
    device = torch.device("cpu")
    # Opencv
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_p2c = np.loadtxt("tests/data/spp_probe2camera.txt")
    test_matrix_p2l = np.linalg.inv(test_matrix_l2c) @ test_matrix_p2c

    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    test_matrix_p2c = np.expand_dims(test_matrix_p2c, 0)
    test_matrix_p2l = np.expand_dims(test_matrix_p2l, 0)

    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)

    opengl_l2c, r_gl, t_gl = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    rp = test_matrix_p2c[:, :3, :3]
    tp = test_matrix_p2c[:, :3, 3]
    test_matrix_p2c_torch_r = torch.from_numpy(rp)
    test_matrix_p2c_torch_t = torch.from_numpy(tp)
    opengl_p2c, rp_gl, tp_gl = vru.opencv_to_opengl(test_matrix_p2c_torch_r, test_matrix_p2c_torch_t, device)

    p2c_pytorch = Transform3d(matrix=opengl_p2c)
    l2c_pytorch = Transform3d(matrix=opengl_l2c)
    p2l_pytorch = p2c_pytorch.compose(l2c_pytorch.inverse()).get_matrix()
    # Testing getting p2l from composed transforms.
    rp2l_gl = p2l_pytorch[:, :3, :3]
    tp2l_gl = p2l_pytorch[:, 3, :3]
    p2l_opencv, _, _ = vru.opengl_to_opencv_p2l(rp2l_gl, tp2l_gl, device)
    slicesampler_p2l_pytorch = vru.p2l_2_slicesampler(p2l_opencv)
    # Get of p2l from numpy
    p2l_from_numpy = torch.from_numpy(test_matrix_p2l)
    slicesampler_p2l_numpy = vru.p2l_2_slicesampler(p2l_from_numpy)
    assert np.allclose(slicesampler_p2l_numpy.numpy(), slicesampler_p2l_pytorch.numpy())

def test_split_cv_hom():
    """
    Test if we split our hom opencv
    matrix we get the same as using
    numpy indexations.
    """
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]

    split_r, split_t = vru.split_opencv_hom_matrix(torch.from_numpy(test_matrix_l2c))
    assert np.allclose(split_r.numpy(), r)
    assert np.allclose(split_t.numpy(), t)

def test_split_gl_hom():
    """
    Test if we split our hom opengl
    matrix we get the same as using
    numpy indexations.
    """
    device = torch.device("cpu")
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M, split_r, split_t = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    split_r_test, split_t_test = vru.split_opengl_hom_matrix(M)
    assert np.allclose(split_r.numpy(), split_r_test.numpy())
    assert np.allclose(split_t.numpy(), split_t_test.numpy())
