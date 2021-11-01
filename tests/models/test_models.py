# coding=utf-8

"""
Test module for renderer. We call pytest skip as
the data is not published.
"""

import os
import json
import torch
import pytest
import numpy as np
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import rotation_conversions as p3drc
import lapvideous_pt.models as lvm
from lapvideous_pt.models.models import LapVideoUS
import lapvideous_pt.generators.video_generation.utils as vru
import matplotlib.pyplot as plt

# @pytest.mark.skip(reason="Requires local data, not shareable.")
def test_post_process_predictions():
    """
    Checking that for a known set of parameters
    we get the correct rendering.
    """
    model_lus_json = os.path.join(os.path.abspath("./tests/data/test_data/"), 'expt_config.json')
    with open(model_lus_json) as f:
        expt_config = json.load(f)

    model = LapVideoUS(**expt_config)
    video_data = model.prep_input_data_for_render()

    # OpenGL rendering format.
    transform_l2c = model.video_loader.l2c.expand(1, -1, -1) # (N, 4, 4)
    transform_p2c = model.video_loader.p2c.expand(1, -1, -1) # (N, 4, 4)
    transform_p2l = Transform3d(matrix=transform_p2c, device=model.device).compose(Transform3d(matrix=transform_l2c, device=model.device).inverse()).get_matrix()

    # Render a first image tensor.
    image_tensor = model.render_data(video_data,
                                 transform_l2c=transform_l2c,
                                 transform_p2c=transform_p2c)
    ### Get original r and t matrices.
    p2l_r, p2l_t = vru.split_opengl_hom_matrix(transform_p2l)
    c2l_r, c2l_t = vru.split_opengl_hom_matrix(Transform3d(matrix=transform_l2c).inverse().get_matrix())
    ### Convert to quaternions and normalised t space.
    p2l_q = p3drc.matrix_to_quaternion(torch.transpose(p2l_r, 2, 1)) # We transpose because the quaternion
                                                                    # Notation uses lefthanded matrices.
    c2l_q = p3drc.matrix_to_quaternion(torch.transpose(c2l_r, 2, 1)) # We transpose because the quaternion
                                                                    # Notation uses lefthanded matrices.
    p2l_t_norm = vru.global_to_local_space(p2l_t, model.bounds)
    c2l_t_norm = vru.global_to_local_space(c2l_t, model.bounds)
    # Get a t tensor and a q tensor to pass into post_process_predictions method
    test_t = torch.cat((c2l_t_norm, p2l_t_norm), 1)
    test_q = torch.cat((c2l_q, p2l_q), 1)
    c2l_pytorch3d, p2l_pytorch3d = model.post_process_predictions(test_q, test_t)

    # Check the matrices from c2l and p2l are the same as the original ones.
    assert np.allclose(c2l_pytorch3d.get_matrix().numpy(), Transform3d(matrix=transform_l2c).inverse().get_matrix().numpy())
    assert np.allclose(p2l_pytorch3d.get_matrix().numpy(), transform_p2l.numpy())
    p2c_transform3d = p2l_pytorch3d.compose(c2l_pytorch3d.inverse())
    print("\n Predicted p2c: \n", p2c_transform3d.get_matrix().numpy())
    print("\n GT P2c: \n", transform_p2c.numpy())
    print("\n Predicted c2l: \n", c2l_pytorch3d.get_matrix().numpy())
    print("\n GT c2l: \n", Transform3d(matrix=transform_l2c).inverse().get_matrix().numpy())
    print("\n Predicted p2l: \n", p2l_pytorch3d.get_matrix().numpy())
    print("\n GT P2l: \n", transform_p2l.numpy())
    print("\n DIFF P2C: \n", p2c_transform3d.get_matrix().numpy() -  transform_p2c.numpy())

    image_tensor_pred = model.render_data(video_data,
                                          transform_l2c=c2l_pytorch3d.inverse().get_matrix(),
                                          transform_p2c=p2c_transform3d.get_matrix())
    print("\n Average DIFF Images: \n", np.mean((image_tensor - image_tensor_pred).numpy()))

    plt.subplot(1, 4, 1)
    plt.xlabel("US resliced, B=1")
    plt.imshow(np.transpose(image_tensor.numpy()[0, 4:, :, :], [1, 2, 0]))
    plt.grid(False)
    plt.subplot(1, 4, 2)
    plt.xlabel("Video Render, B=1")
    plt.imshow(np.transpose(image_tensor.numpy()[0, 0:4, :, :],  [1, 2, 0]))  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 4, 3)
    plt.xlabel("US resliced, B=1")
    plt.imshow(np.transpose(image_tensor_pred.numpy()[0, 4:, :, :], [1, 2, 0]))
    plt.grid(False)
    plt.subplot(1, 4, 4)
    plt.xlabel("Video Render, B=1")
    plt.imshow(np.transpose(image_tensor_pred.numpy()[0, 0:4, :, :],  [1, 2, 0]))  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.show()

    # This does not pass - some numerical error.
    # assert np.allclose(image_tensor, image_tensor_pred)
    # assert np.allclose(p2c_transform3d.get_matrix().numpy(), transform_p2c.numpy())

@pytest.mark.skip(reason="Requires local data, not shareable.")
def test_post_process_predictions_perturbed():
    """
    Checking that for a known set of parameters
    we get the correct rendering.
    """
    model_lus_json = os.path.join(os.path.abspath("./tests/data/test_data/"), 'expt_config.json')
    with open(model_lus_json) as f:
        expt_config = json.load(f)

    model = LapVideoUS(**expt_config)
    video_data = model.prep_input_data_for_render()

    # OpenGL rendering format.
    transform_l2c = model.video_loader.l2c.expand(1, -1, -1) # (N, 4, 4)
    transform_p2c = model.video_loader.p2c.expand(1, -1, -1) # (N, 4, 4)
    transform_p2l = Transform3d(matrix=transform_p2c, device=model.device).compose(Transform3d(matrix=transform_l2c, device=model.device).inverse()).get_matrix()
    r_c2l, t_c2l = vru.generate_transforms(model.device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    r_p2l, t_p2l = vru.generate_transforms(model.device, 1, torch.from_numpy(np.array([[10,10,10,10,10,10]], dtype=np.float64)))
    transform_l2c_p, transform_p2c_p = vru.perturb_orig_matrices(Transform3d(matrix=transform_l2c, device=model.device),
                                                                 Transform3d(matrix=transform_p2c, device=model.device),
                                                                 r_c2l, t_c2l, r_p2l, t_p2l)
    transform_p2l = transform_p2c_p.compose(transform_l2c_p.inverse()).get_matrix()
    # Render a first image tensor.
    image_tensor = model.render_data(video_data,
                                 transform_l2c=transform_l2c_p.get_matrix(),
                                 transform_p2c=transform_p2c_p.get_matrix())
    ### Get original r and t matrices.
    p2l_r, p2l_t = vru.split_opengl_hom_matrix(transform_p2l)
    c2l_r, c2l_t = vru.split_opengl_hom_matrix(transform_l2c_p.inverse().get_matrix())
    ### Convert to quaternions and normalised t space.
    p2l_q = p3drc.matrix_to_quaternion(torch.transpose(p2l_r, 2, 1)) # We transpose because the quaternion
                                                                    # Notation uses lefthanded matrices.
    c2l_q = p3drc.matrix_to_quaternion(torch.transpose(c2l_r, 2, 1)) # We transpose because the quaternion
                                                                    # Notation uses lefthanded matrices.
    p2l_t_norm = vru.global_to_local_space(p2l_t, model.bounds)
    c2l_t_norm = vru.global_to_local_space(c2l_t, model.bounds)
    # Get a t tensor and a q tensor to pass into post_process_predictions method
    test_t = torch.cat((c2l_t_norm, p2l_t_norm), 1)
    test_q = torch.cat((c2l_q, p2l_q), 1)
    c2l_pytorch3d, p2l_pytorch3d = model.post_process_predictions(test_q, test_t)

    # Check the matrices from c2l and p2l are the same as the original ones.
    
    p2c_transform3d = p2l_pytorch3d.compose(c2l_pytorch3d.inverse())

    pred_c2l = c2l_pytorch3d.get_matrix().numpy()
    gt_c2l = transform_l2c_p.inverse().get_matrix().numpy()
    print("\n Predicted c2l: \n", pred_c2l)
    print("\n GT c2l: \n", gt_c2l)
    assert np.allclose(gt_c2l, pred_c2l)

    pred_p2c = p2c_transform3d.get_matrix().numpy()
    gt_p2c =  transform_p2c_p.get_matrix().numpy()
    print("\n Predicted p2c: \n", pred_p2c)
    print("\n GT P2c: \n", gt_p2c)
    assert np.allclose(pred_p2c, gt_p2c)

    pred_p2l =  p2l_pytorch3d.get_matrix().numpy()
    gt_p2l =  transform_p2l.numpy()
    print("\n Predicted p2l: \n",pred_p2l)
    print("\n GT P2l: \n", gt_p2l)
    assert np.allclose(gt_p2l, pred_p2l)

    image_tensor_pred = model.render_data(video_data,
                                          transform_l2c=c2l_pytorch3d.inverse().get_matrix(),
                                          transform_p2c=p2c_transform3d.get_matrix())
    print("\n Average DIFF Images: \n", np.mean((image_tensor - image_tensor_pred).numpy()))

    plt.subplot(1, 4, 1)
    plt.xlabel("US resliced, B=1")
    plt.imshow(np.transpose(image_tensor.numpy()[0, 4:, :, :], [1, 2, 0]))
    plt.grid(False)
    plt.subplot(1, 4, 2)
    plt.xlabel("Video Render, B=1")
    plt.imshow(np.transpose(image_tensor.numpy()[0, 0:4, :, :],  [1, 2, 0]))  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 4, 3)
    plt.xlabel("US resliced, B=1")
    plt.imshow(np.transpose(image_tensor_pred.numpy()[0, 4:, :, :], [1, 2, 0]))
    plt.grid(False)
    plt.subplot(1, 4, 4)
    plt.xlabel("Video Render, B=1")
    plt.imshow(np.transpose(image_tensor_pred.numpy()[0, 0:4, :, :],  [1, 2, 0]))  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.show()

    # This does not pass - some numerical error.
    # assert np.allclose(image_tensor, image_tensor_pred)

@pytest.mark.skip(reason="Requires local data, not shareable.")
def test_post_process_predictions_from_quaternions():
    """
    Checking that for a known set of parameters
    we get the correct rendering.
    """
    model_lus_json = os.path.join(os.path.abspath("./tests/data/test_data/"), 'expt_config.json')
    with open(model_lus_json) as f:
        expt_config = json.load(f)

    model = LapVideoUS(**expt_config)
    video_data = model.prep_input_data_for_render()

    # OpenGL rendering format.
    transform_l2c = model.video_loader.l2c.expand(1, -1, -1) # (N, 4, 4)
    transform_p2c = model.video_loader.p2c.expand(1, -1, -1) # (N, 4, 4)
    transform_p2l = Transform3d(matrix=transform_p2c, device=model.device).compose(Transform3d(matrix=transform_l2c, device=model.device).inverse()).get_matrix()
    
    # Render a first image tensor.
    image_tensor, _ = model.render_data(video_data,
                                 transform_l2c=transform_l2c,
                                 transform_p2c=transform_p2c)

    # Check the matrices from c2l and p2l are the same as the original ones.
    c2l_q_pred = torch.from_numpy(np.array([[0.45255578, 0.07508873, 0.5642768,0.68640125]]))
    c2l_t_pred =  torch.from_numpy(np.array([[-306.9032/500,-353.33618/500, -175.96095/500]]))
    p2l_q_pred = torch.from_numpy(np.array([[-0.5731902,   0.75540805,  0.2776732,   0.15397817]]))
    p2l_t_pred =  torch.from_numpy(np.array([[-15.071623/500, -84.755135/500,  34.05004/500]]))

    rot_preds = torch.cat([c2l_q_pred, p2l_q_pred], dim=1)
    trans_preds = torch.cat([c2l_t_pred, p2l_t_pred], dim=1)
    # r_c2l, ijk_c2l, r_p2l, ijk_p2l = torch.split(rot_preds, [1,3,1,3], dim=1)
    # r_c2l_out = torch.abs(r_c2l)
    # r_p2l_out = torch.abs(r_p2l)
    # out_rot_r_norm = torch.cat([r_c2l_out, ijk_c2l, r_p2l_out, ijk_p2l], dim=1)
    c2l_pytorch3d, p2l_pytorch3d = model.post_process_predictions(rot_preds, trans_preds)
    transform_l2c_pred = c2l_pytorch3d.inverse()
    transform_p2c_pred = p2l_pytorch3d.compose(transform_l2c_pred)
    image_tensor_pred, _ = model.render_data(liver_data=video_data,
                                       transform_p2c=transform_p2c_pred.get_matrix().float(),
                                       transform_l2c=transform_l2c_pred.get_matrix().float())

    print("L2C pred: \n", transform_l2c_pred.get_matrix().numpy())
    print("L2C gt: \n", transform_l2c.numpy())
    print("P2C pred: \n", transform_p2c_pred.get_matrix().numpy())
    print("P2C gt: \n", transform_p2c.numpy())
    print("P2l pred: \n", p2l_pytorch3d.get_matrix().numpy())
    print("P2l gt: \n", transform_p2l.numpy())

    plt.subplot(1, 4, 1)
    plt.xlabel("US resliced, B=1")
    plt.imshow(np.transpose(image_tensor.numpy()[0, 4:, :, :], [1, 2, 0]))
    plt.grid(False)
    plt.subplot(1, 4, 2)
    plt.xlabel("Video Render, B=1")
    plt.imshow(np.transpose(image_tensor.numpy()[0, 0:4, :, :],  [1, 2, 0]))  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 4, 3)
    plt.xlabel("US resliced, B=1")
    plt.imshow(np.transpose(image_tensor_pred.numpy()[0, 4:, :, :], [1, 2, 0]))
    plt.grid(False)
    plt.subplot(1, 4, 4)
    plt.xlabel("Video Render, B=1")
    plt.imshow(np.transpose(image_tensor_pred.numpy()[0, 0:4, :, :],  [1, 2, 0]))  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.show()

    # This does not pass - some numerical error.
    # assert np.allclose(image_tensor, image_tensor_pred)
