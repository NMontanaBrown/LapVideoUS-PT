# coding=utf-8

"""
Testing transformations
"""

import os
import pytest
import numpy as np
import torch
from torch.functional import split
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import rotation_conversions as p3drc
import lapvideous_pt.generators.video_generation.utils as vru
from sksurgeryvtk.utils.matrix_utils import create_matrix_from_list 

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

def test_split_join_gl_hom():
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
    join_M = vru.cat_opengl_hom_matrix(split_r_test, split_t_test, device)
    join_M_orig = vru.cat_opengl_hom_matrix(split_r, split_t, device)
    assert np.allclose(join_M, M)
    assert np.allclose(M, join_M_orig)

def test_split_join_cv_hom():
    """
    Test if we split our hom opencv
    matrix we get the same as using
    numpy indexations.
    """
    device = torch.device("cpu")
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]

    split_r, split_t = vru.split_opencv_hom_matrix(torch.from_numpy(test_matrix_l2c))
    join_M = vru.cat_opencv_hom_matrix(split_r, split_t, device)

    assert np.allclose(join_M, test_matrix_l2c)

def test_gen_random_params_index():
    """
    Test that we get random numbers generated
    at appropriate indices.
    """
    device = torch.device("cpu")
    generated_values = vru.generate_random_params_index(device, 4, [0, 3], [10, 10], [20,20])
    split_values = torch.split(generated_values, [1,1,1,1,1,1], 1)
    # Check indices not at 0, 3 are zeross
    assert np.allclose(split_values[1].numpy(), split_values[-1].numpy())

def test_gen_ordered_params_index():
    """
    Test we get an ordered list as expected.
    """
    device = torch.device("cpu")
    generated_values = vru.generate_ordered_params_index(device, 10, [0], [-10], [0])
    split_values, _ = torch.split(generated_values, [1,5], 1)
    assert np.allclose(split_values.numpy(), np.expand_dims(np.linspace(-10, 0, 10), 1))

def test_generate_transforms():
    """
    Test we get can call and get a series of transforms
    """
    device = torch.device("cpu")
    generated_values = vru.generate_transforms(device, 3, None)
    return True

@pytest.mark.parametrize("transform_list_r, transform_list_t, transform_list_both", [
                         ([10,10,10,0,0,0],[0,0,0,10,10,10], [10,10,10,10,10,10]),
                         ([-10,-10,-10,0,0,0],[0,0,0, -10,-10,-10], [-10,-10,-10,-10,-10,-10]) 
])
def test_generate_transforms_regression(transform_list_r, transform_list_t,transform_list_both):
    """
    Check that by using vtk utils to generate some
    transforms we get the same as with pytorch
    """
    device = torch.device("cpu")
    # Generate a rotation of 10 degrees
    # And a translation of 10 mm in all directions.
    mat_c2l_r = create_matrix_from_list(transform_list_r,
                                       is_in_radians=False)
    mat_c2l_t = create_matrix_from_list(transform_list_t,
                                       is_in_radians=False)
    rot_c2l, t_c2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([transform_list_both], dtype=np.float64)))
    rot_c2l, _ = vru.split_opengl_hom_matrix(rot_c2l.get_matrix())
    _, t_c2l = vru.split_opengl_hom_matrix(t_c2l.get_matrix())

    _, rot_c2l_opencv, t_c2l_opencv = vru.opengl_to_opencv_p2l(rot_c2l, t_c2l, device)
    assert np.allclose(rot_c2l_opencv.numpy(), mat_c2l_r[:3, :3])
    assert np.allclose(t_c2l_opencv.numpy(), mat_c2l_t[:3, 3])

@pytest.mark.parametrize("transform_list_t, transform_list_both", [
                         ([0,0,0,10,10,10], [0,0,0,10,10,10]),
                         ([0,0,0, -10,-10,-10], [0,0,0,-10,-10,-10]) 
])
def test_perturb_orig_matrices_regression_t_l2c(transform_list_t, transform_list_both):
    """
    Testing that if we apply a translation to the original matrix we can
    recover the same as sksurgery-vtk
    """
    device = torch.device("cpu")
    # L2c
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_c2l = np.linalg.inv(test_matrix_l2c)
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M_l2c, _, _ = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    # c2l perturb using sksurgery vtk
    mat_c2l_t = create_matrix_from_list(transform_list_t,
                                       is_in_radians=False)
    transformed_c2l_numpy = mat_c2l_t @ test_matrix_c2l
    _, t_c2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([transform_list_both], dtype=np.float64)))
    # Transform c2l using rot
    l2c_transform3d = Transform3d(matrix=M_l2c.double(), device=device)
    permuted_l2c = Transform3d(device=device, dtype=torch.double).compose(
                                                                  l2c_transform3d.inverse(),
                                                                  Transform3d(matrix=t_c2l.get_matrix().double())
                                                                  ).inverse()
    # Should give the same matrix
    permuted_l2c_r, permuted_l2c_t = vru.split_opengl_hom_matrix(permuted_l2c.get_matrix())
    M_l2c_permuted_cv,_,_ = vru.opengl_to_opencv(permuted_l2c_r, permuted_l2c_t, device)
    print("\nTransform l2c from opengl to opencv: \n", M_l2c_permuted_cv.numpy())
    print("\nTransform l2c using vtk: \n", np.linalg.inv(transformed_c2l_numpy))
    assert np.allclose(M_l2c_permuted_cv.numpy(), np.linalg.inv(transformed_c2l_numpy))

def test_perturb_orig_matrices_regression_t_p2l():
    """
    Testing that if we apply a translation to the original matrix we can
    recover the same as sksurgery-vtk
    """
    device = torch.device("cpu")
    # L2c
    test_matrix_p2c = np.loadtxt("tests/data/spp_probe2camera.txt")
    test_matrix_p2c = np.expand_dims(test_matrix_p2c, 0)
    r = test_matrix_p2c[:, :3, :3]
    t = test_matrix_p2c[:, :3, 3]
    test_matrix_p2c_torch_r = torch.from_numpy(r)
    test_matrix_p2c_torch_t = torch.from_numpy(t)
    M_p2c, _, _ = vru.opencv_to_opengl(test_matrix_p2c_torch_r, test_matrix_p2c_torch_t, device)
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_c2l = np.linalg.inv(test_matrix_l2c)
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M_l2c, _, _ = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    # c2l perturb using sksurgery vtk
    mat_c2l_t = create_matrix_from_list([0,0,0,10,10,10],
                                       is_in_radians=False)
    transformed_c2l_numpy = mat_c2l_t @ test_matrix_c2l
    perturbed_p2l = transformed_c2l_numpy @ test_matrix_p2c
    _, t_c2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    # Transform c2l using rot
    l2c_transform3d = Transform3d(matrix=M_l2c.double(), device=device)
    permuted_l2c = Transform3d(device=device, dtype=torch.double).compose(
                                                                  l2c_transform3d.inverse(),
                                                                  Transform3d(matrix=t_c2l.get_matrix().double())
                                                                  ).inverse()
    # Transform p2c using rot
    p2c_transform3d = Transform3d(matrix=M_p2c.double(), device=device)
    permuted_p2l = Transform3d(device=device, dtype=torch.double).compose(
                                                                  p2c_transform3d,
                                                                  permuted_l2c.inverse(),
                                                                  )
    # Should give the same matrix
    permuted_p2l_r, permuted_p2l_t = vru.split_opengl_hom_matrix(permuted_p2l.get_matrix())
    M_p2l_permuted_cv,_,_ = vru.opengl_to_opencv_p2l(permuted_p2l_r, permuted_p2l_t, device)
    print("\nTransform p2l from opengl to opencv: \n", M_p2l_permuted_cv.numpy())
    print("\nTransform p2l using vtk: \n", perturbed_p2l)
    assert np.allclose(M_p2l_permuted_cv.numpy(), perturbed_p2l)

def test_perturb_orig_matrices_regression_r_p2l():
    """
    Testing that if we apply a translation to the original matrix we can
    recover the same as sksurgery-vtk
    """
    device = torch.device("cpu")
    # L2c
    test_matrix_p2c = np.loadtxt("tests/data/spp_probe2camera.txt")
    test_matrix_p2c = np.expand_dims(test_matrix_p2c, 0)
    r = test_matrix_p2c[:, :3, :3]
    t = test_matrix_p2c[:, :3, 3]
    test_matrix_p2c_torch_r = torch.from_numpy(r)
    test_matrix_p2c_torch_t = torch.from_numpy(t)
    M_p2c, _, _ = vru.opencv_to_opengl(test_matrix_p2c_torch_r, test_matrix_p2c_torch_t, device)
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_c2l = np.linalg.inv(test_matrix_l2c)
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M_l2c, _, _ = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    # c2l perturb using sksurgery vtk
    mat_c2l_r = create_matrix_from_list([10,10,10,0,0,0],
                                       is_in_radians=False)
    transformed_c2l_numpy = mat_c2l_r @ test_matrix_c2l
    perturbed_p2l = transformed_c2l_numpy @ test_matrix_p2c
    r_c2l, _ = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    # Transform c2l using rot
    l2c_transform3d = Transform3d(matrix=M_l2c.double(), device=device)
    permuted_l2c = Transform3d(device=device, dtype=torch.double).compose(
                                                                  l2c_transform3d.inverse(),
                                                                  Transform3d(matrix=r_c2l.get_matrix().double())
                                                                  ).inverse()
    # Transform p2c using rot
    p2c_transform3d = Transform3d(matrix=M_p2c.double(), device=device)
    permuted_p2l = Transform3d(device=device, dtype=torch.double).compose(
                                                                  p2c_transform3d,
                                                                  permuted_l2c.inverse(),
                                                                  )
    # Should give the same matrix
    permuted_p2l_r, permuted_p2l_t = vru.split_opengl_hom_matrix(permuted_p2l.get_matrix())
    M_p2l_permuted_cv,_,_ = vru.opengl_to_opencv_p2l(permuted_p2l_r, permuted_p2l_t, device)
    print("\nTransform p2l from opengl to opencv: \n", M_p2l_permuted_cv.numpy())
    print("\nTransform p2l using vtk: \n", perturbed_p2l)
    assert np.allclose(M_p2l_permuted_cv.numpy(), perturbed_p2l)

def test_perturb_orig_matrices_regression_r_l2c():
    """
    Testing that if we apply a rotation to the original matrix we can
    recover the same as sksurgery-vtk
    """
    device = torch.device("cpu")
    # L2c
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_c2l = np.linalg.inv(test_matrix_l2c)
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M_l2c, _, _ = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    # c2l perturb using sksurgery vtk
    mat_c2l_r = create_matrix_from_list([10,10,10,0,0,0],
                                       is_in_radians=False)
    transformed_c2l_numpy = mat_c2l_r @ test_matrix_c2l
    r_c2l, _ = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    # Transform c2l using rot and 2
    l2c_transform3d = Transform3d(matrix=M_l2c.double(), device=device)
    permuted_l2c = Transform3d(device=device, dtype=torch.double).compose(
                                                                  l2c_transform3d.inverse(),
                                                                  Transform3d(matrix=r_c2l.get_matrix().double())
                                                                  ).inverse()
    permuted_l2c_r, permuted_l2c_t = vru.split_opengl_hom_matrix(permuted_l2c.get_matrix())
    M_l2c_permuted_cv,_,_ = vru.opengl_to_opencv(permuted_l2c_r, permuted_l2c_t, device)
    print("\nTransform l2c from opengl to opencv: \n", M_l2c_permuted_cv.numpy())
    print("\nTransform l2c using vtk: \n", np.linalg.inv(transformed_c2l_numpy))
    assert np.allclose(M_l2c_permuted_cv.numpy(), np.linalg.inv(transformed_c2l_numpy))

def test_perturb_orig_matrices_regression_rt_l2c():
    """
    Testing that if we apply a translation to the original matrix we can
    recover the same as sksurgery-vtk
    """
    device = torch.device("cpu")
    # L2c
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_c2l = np.linalg.inv(test_matrix_l2c)
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M_l2c, _, _ = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    # c2l perturb using sksurgery vtk
    mat_c2l_r = create_matrix_from_list([10,10,10,0,0,0],
                                       is_in_radians=False)
    mat_c2l_t = create_matrix_from_list([0,0,0,10,10,10],
                                       is_in_radians=False)
    transformed_c2l_numpy = mat_c2l_t @ mat_c2l_r @ test_matrix_c2l 
    r_c2l, t_c2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    # Transform c2l using rot and 2
    l2c_transform3d = Transform3d(matrix=M_l2c.double(), device=device)
    permuted_l2c = Transform3d(device=device, dtype=torch.double).compose(
                                                                  l2c_transform3d.inverse(),
                                                                  Transform3d(matrix=r_c2l.get_matrix().double()),
                                                                  Transform3d(matrix=t_c2l.get_matrix().double()),
                                                                  ).inverse()

    permuted_l2c_r, permuted_l2c_t = vru.split_opengl_hom_matrix(permuted_l2c.get_matrix())
    M_l2c_permuted_cv,_,_ = vru.opengl_to_opencv(permuted_l2c_r, permuted_l2c_t, device)
    print("\nTransform l2c from opengl to opencv: \n", M_l2c_permuted_cv.numpy())
    print("\nTransform l2c using vtk: \n", np.linalg.inv(transformed_c2l_numpy))
    assert np.allclose(M_l2c_permuted_cv.numpy(), np.linalg.inv(transformed_c2l_numpy))

def test_perturb_orig_matrices_regression_rt_p2l():
    """
    Testing that if we apply a translation to the original matrix we can
    recover the same as sksurgery-vtk
    """
    device = torch.device("cpu")
    # L2c
    test_matrix_p2c = np.loadtxt("tests/data/spp_probe2camera.txt")
    test_matrix_p2c = np.expand_dims(test_matrix_p2c, 0)
    r = test_matrix_p2c[:, :3, :3]
    t = test_matrix_p2c[:, :3, 3]
    test_matrix_p2c_torch_r = torch.from_numpy(r)
    test_matrix_p2c_torch_t = torch.from_numpy(t)
    M_p2c, _, _ = vru.opencv_to_opengl(test_matrix_p2c_torch_r, test_matrix_p2c_torch_t, device)
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_c2l = np.linalg.inv(test_matrix_l2c)
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M_l2c, _, _ = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    # c2l perturb using sksurgery vtk
    mat_c2l_r = create_matrix_from_list([10,10,10,10,10,10],
                                       is_in_radians=False)
    transformed_c2l_numpy = mat_c2l_r @ test_matrix_c2l
    perturbed_p2l = transformed_c2l_numpy @ test_matrix_p2c
    r_c2l, t_c2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    # Transform c2l using rot
    l2c_transform3d = Transform3d(matrix=M_l2c.double(), device=device)
    permuted_l2c = Transform3d(device=device, dtype=torch.double).compose(
                                                                  l2c_transform3d.inverse(),
                                                                  Transform3d(matrix=r_c2l.get_matrix().double()),
                                                                  Transform3d(matrix=t_c2l.get_matrix().double())
                                                                  ).inverse()
    # Transform p2c using rot
    p2c_transform3d = Transform3d(matrix=M_p2c.double(), device=device)
    permuted_p2l = Transform3d(device=device, dtype=torch.double).compose(
                                                                  p2c_transform3d,
                                                                  permuted_l2c.inverse(),
                                                                  )
    # Should give the same matrix
    permuted_p2l_r, permuted_p2l_t = vru.split_opengl_hom_matrix(permuted_p2l.get_matrix())
    M_p2l_permuted_cv,_,_ = vru.opengl_to_opencv_p2l(permuted_p2l_r, permuted_p2l_t, device)
    print("\nTransform p2l from opengl to opencv: \n", M_p2l_permuted_cv.numpy())
    print("\nTransform p2l using vtk: \n", perturbed_p2l)
    assert np.allclose(M_p2l_permuted_cv.numpy(), perturbed_p2l)


def test_perturb_orig_matrices_regression_rt_combined():
    """
    Testing that if we apply two sets of transformations
    to the original p2l and c2l at the same time
    recover the same as sksurgery-vtk.
    """
    device = torch.device("cpu")
    # L2c
    test_matrix_p2c = np.loadtxt("tests/data/spp_probe2camera.txt")
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_p2l = np.linalg.inv(test_matrix_l2c) @ test_matrix_p2c

    test_matrix_c2l = np.linalg.inv(test_matrix_l2c)
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M_l2c, _, _ = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)

    # c2l perturb using sksurgery vtk
    mat_c2l_r = create_matrix_from_list([10,10,10,10,10,10],
                                       is_in_radians=False)
    transformed_c2l_numpy = mat_c2l_r @ test_matrix_c2l

    test_matrix_p2l = np.expand_dims(test_matrix_p2l, 0)
    r = test_matrix_p2l[:, :3, :3]
    t = test_matrix_p2l[:, :3, 3]
    test_matrix_p2l_torch_r = torch.from_numpy(r)
    test_matrix_p2l_torch_t = torch.from_numpy(t)
    M_p2l, _, _ = vru.opencv_to_opengl_p2l(test_matrix_p2l_torch_r, test_matrix_p2l_torch_t, device)
    # c2l perturb using sksurgery vtk
    mat_p2l_r = create_matrix_from_list([10,10,10,10,10,10],
                                       is_in_radians=False)
    transformed_p2l_numpy = mat_p2l_r @ test_matrix_p2l

    perturbed_p2c = np.linalg.inv(transformed_c2l_numpy) @ transformed_p2l_numpy
    r_c2l, t_c2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    r_p2l, t_p2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    # Transform c2l using rot
    l2c_transform3d = Transform3d(matrix=M_l2c.double(), device=device)
    permuted_l2c = Transform3d(device=device, dtype=torch.double).compose(
                                                                  l2c_transform3d.inverse(),
                                                                  Transform3d(matrix=r_c2l.get_matrix().double()),
                                                                  Transform3d(matrix=t_c2l.get_matrix().double())
                                                                  ).inverse()
    # Transform p2l using rot
    p2l_transform3d = Transform3d(matrix=M_p2l.double(), device=device)
    permuted_p2l = Transform3d(device=device, dtype=torch.double).compose(
                                                                  p2l_transform3d,
                                                                  Transform3d(matrix=r_p2l.get_matrix().double()),
                                                                  Transform3d(matrix=t_p2l.get_matrix().double())
                                                                  )
    # Get p2c
    p2c_transform3d = Transform3d(device=device, dtype=torch.double).compose(
                                                                    permuted_p2l,
                                                                    permuted_l2c
                                                                    )
    # Should give the same matrix
    permuted_p2c_r, permuted_p2c_t = vru.split_opengl_hom_matrix(p2c_transform3d.get_matrix())
    M_p2c_permuted_cv,_,_ = vru.opengl_to_opencv(permuted_p2c_r, permuted_p2c_t, device)
    print("\nTransform p2c from opengl to opencv: \n", M_p2c_permuted_cv.numpy())
    print("\nTransform p2c using vtk: \n", perturbed_p2c)
    assert np.allclose(M_p2c_permuted_cv.numpy(), perturbed_p2c)

def test_perturb_orig_matrices():
    """
    Testing that if we use the function
    perturb_orig_matrices, we get the same
    result as if we were using vtk utils
    to generate some matrices.
    """
    device = torch.device("cpu")
    r_c2l, t_c2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    r_p2l, t_p2l = vru.generate_transforms(device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    test_matrix_p2c = np.loadtxt("tests/data/spp_probe2camera.txt")
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_p2l = np.linalg.inv(test_matrix_l2c) @ test_matrix_p2c

    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r).double()
    test_matrix_l2c_torch_t = torch.from_numpy(t).double()
    M_l2c, _, _ = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    l2c_transform3d = Transform3d(matrix=M_l2c.double(), device=device)
    
    test_matrix_p2c = np.expand_dims(test_matrix_p2c, 0)
    r = test_matrix_p2c[:, :3, :3]
    t = test_matrix_p2c[:, :3, 3]
    test_matrix_p2c_torch_r = torch.from_numpy(r).double()
    test_matrix_p2c_torch_t = torch.from_numpy(t).double()
    M_p2c, _, _ = vru.opencv_to_opengl(test_matrix_p2c_torch_r, test_matrix_p2c_torch_t, device)
    p2c_transform3d = Transform3d(matrix=M_p2c.double(), device=device)

    transform_l2c_p, transform_p2c_p = vru.perturb_orig_matrices(l2c_transform3d, p2c_transform3d, r_c2l, t_c2l, r_p2l, t_p2l)
    permuted_l2c_r, permuted_l2c_t = vru.split_opengl_hom_matrix(transform_l2c_p.get_matrix())
    M_l2c_permuted_cv,_,_ = vru.opengl_to_opencv(permuted_l2c_r, permuted_l2c_t, device)
    permuted_p2c_r, permuted_p2c_t = vru.split_opengl_hom_matrix(transform_p2c_p.get_matrix())
    M_p2c_permuted_cv,_,_ = vru.opengl_to_opencv(permuted_p2c_r, permuted_p2c_t, device)

    mat_c2l_r = create_matrix_from_list([10,10,10,10,10,10],
                                       is_in_radians=False)
    transformed_c2l_numpy = mat_c2l_r @ np.linalg.inv(test_matrix_l2c)
    mat_p2l_r = create_matrix_from_list([10,10,10,10,10,10],
                                       is_in_radians=False)
    transformed_p2l_numpy = mat_p2l_r @ test_matrix_p2l

    perturbed_p2c = np.linalg.inv(transformed_c2l_numpy) @ transformed_p2l_numpy
    assert np.allclose(M_l2c_permuted_cv.numpy(), np.linalg.inv(transformed_c2l_numpy))
    assert np.allclose(M_p2c_permuted_cv.numpy(), perturbed_p2c)

def test_quaternion_conversion_l2c():
    """
    Testing what happens with preds after converting
    from quaternion to final matrix.
    """
    from scipy.spatial.transform import Rotation as R
    device = torch.device("cpu")
    # L2c
    test_matrix_l2c = np.loadtxt("tests/data/spp_liver2camera.txt")
    test_matrix_c2l = np.linalg.inv(test_matrix_l2c)
    test_matrix_l2c = np.expand_dims(test_matrix_l2c, 0)
    r = test_matrix_l2c[:, :3, :3]
    t = test_matrix_l2c[:, :3, 3]
    test_matrix_l2c_torch_r = torch.from_numpy(r)
    test_matrix_l2c_torch_t = torch.from_numpy(t)
    M_l2c, l2c_r, l2c_t = vru.opencv_to_opengl(test_matrix_l2c_torch_r, test_matrix_l2c_torch_t, device)
    M_l2c_cv, l2c_r_cv, l2c_t_cv = vru.opengl_to_opencv(l2c_r, l2c_t, device)

    # Convert to quaternions and back, to check same matrix is returned.
    l2c_rq = p3drc.matrix_to_quaternion(l2c_r) # OpenGL
    l2c_rq_cv = p3drc.matrix_to_quaternion(l2c_r_cv) # OpenCV
    l2c_rqr_cv = p3drc.quaternion_to_matrix(l2c_rq_cv)
    l2c_rqr_gl = p3drc.quaternion_to_matrix(l2c_rq)

    # This should be internally consistent.
    assert np.allclose(test_matrix_l2c[0, :3, :3], l2c_rqr_cv.numpy())
    # This should be internally consistent.
    assert np.allclose(l2c_r.numpy(), l2c_rqr_gl)

def test_quaternion_unique_solutions():
    """
    Test function that returns unique quaternions constrained
    to the positive hemisphere.
    """
    # Generate a randoom set of quaternions
    random_quaternions = p3drc.random_quaternions(5)
    # Get the other solution by -q
    inv_quaternions = -random_quaternions.clone()
    rand_quat_rot = p3drc.quaternion_to_matrix(random_quaternions)
    inv_quat_rot = p3drc.quaternion_to_matrix(inv_quaternions)
    # Should be the same, as the rotation matrix has the same solution
    # for both sets of quaternions
    assert np.allclose(rand_quat_rot.numpy(), inv_quat_rot.numpy())

    # use our constrain quaternion hemisphere function to convert all
    # quats to a hemisphere
    inv_quaternions_inv = vru.constrain_quat_hemisphere(random_quaternions)
    random_quaternions_inv = vru.constrain_quat_hemisphere(inv_quaternions)
    # Should give the same vectors
    assert np.allclose(inv_quaternions_inv.numpy(), random_quaternions_inv.numpy())
    # Should give the same rotation solution
    assert np.allclose(p3drc.quaternion_to_matrix(inv_quaternions_inv).numpy(),
                       p3drc.quaternion_to_matrix(random_quaternions_inv).numpy())
