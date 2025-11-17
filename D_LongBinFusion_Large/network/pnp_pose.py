'''
Code adapted from
 https://github.com/fangchangma/self-supervised-depth-completion/blob/master/dataloaders/pose_estimator.py
'''
# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv2d, pack_cam_feat, unpack_cam_feat
#from .volumetric_fusionnet import VFNet
from external.layers import ResnetEncoder, PoseDecoder

import cv2
import numpy as np
import os
#os.environ["THESEUS_DISABLE_VMAP"] = "1"
import theseus as th
from theseus.core.cost_function import AutogradMode

def safe_divide(numerator, denominator, eps=1e-8):
    """안전한 나눗셈 함수"""
    return numerator / torch.clamp(denominator, min=eps)

def project_points(pt3ds, T34, K):
    pt3ds_h = torch.cat([pt3ds, torch.ones_like(pt3ds[:,:, :1])], dim=2)
    cam_points = torch.matmul(T34, pt3ds_h.permute(0, 2, 1))
    img_points = torch.matmul(K[:, :3, :3], cam_points).permute(0, 2, 1)
    
    # 안전한 나눗셈 사용
    img_points = safe_divide(img_points, img_points[:, :, 2:])
    return img_points[:, :, :2]


def error_fn(optim_vars, aux_vars):
    T_cw = optim_vars[0]
    obj_points = aux_vars[0]
    img_points = aux_vars[1]
    K = aux_vars[2]
    proj_img_points = project_points(obj_points.tensor, T_cw.tensor, K.tensor) # access tensors inside theseus variables using .tensor.
    error = proj_img_points - img_points.tensor
    error = torch.sqrt(torch.sum(error**2, dim=2)) # do not average values. maintain second dimension. [1, 80]
    return error

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def convert_2d_to_3d(u, v, z, K):
    v0 = K[1][2]
    u0 = K[0][2]
    fy = K[1][1]
    fx = K[0][0]
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy
    return (x, y, z)


def feature_match(img1, img2):
    r''' Find features on both images and match them pairwise
   '''
    max_n_features = 1000
    # max_n_features = 500
    use_flann = False  # better not use flann

    detector = cv2.SIFT_create(max_n_features)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    if (des1 is None) or (des2 is None):
        return [], []
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    if use_flann:
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    else:
        matcher = cv2.DescriptorMatcher().create('BruteForce')
        matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2


def get_pose_pnp(rgb_curr, rgb_near, depth_curr, K):
    gray_curr = rgb2gray(rgb_curr).astype(np.uint8)
    gray_near = rgb2gray(rgb_near).astype(np.uint8)

    pts2d_curr, pts2d_near = feature_match(gray_curr,
                                           gray_near)  # feature matching

    # dilation of depth
    kernel = np.ones((4, 4), np.uint8)
    depth_curr_dilated = cv2.dilate(depth_curr, kernel)

    # extract 3d pts
    pts3d_curr = []
    pts2d_near_filtered = []  # keep only feature points with depth in the current frame
    for i, pt2d in enumerate(pts2d_curr):
        # print(pt2d)
        u, v = pt2d[0], pt2d[1]
        z = depth_curr_dilated[v, u]
        if z > 0:
            xyz_curr = convert_2d_to_3d(u, v, z, K)
            pts3d_curr.append(xyz_curr)
            pts2d_near_filtered.append(pts2d_near[i])

    pts3d_curr = np.expand_dims(np.array(pts3d_curr).astype(np.float32), axis=1)
    pts2d_near_filtered = np.expand_dims(np.array(pts2d_near_filtered).astype(np.float32), axis=1)

    same_length = pts3d_curr.shape[0] == pts2d_near_filtered.shape[0]
    # the minimal number of points accepted by solvePnP is 4:
    required_count = pts3d_curr.shape[0] >= 4

    if same_length and required_count:
        
        '''
        OpenCV에서 solvePnP()의 반환값
        - 타겟 프레임 좌표계에서 본, 현재 프레임 3D 포인트들의 pose 
            => 미래 좌표계에서 본 현재 카메라의 pose
            => 미래 -> 현재 변환
        - "현재에서 미래로 가는 pose"를 얻고 싶다면, 위 pose를 역으로 변환
        '''

        flag = cv2.SOLVEPNP_EPNP
        if pts3d_curr.shape[0] == 4:
            flag = cv2.SOLVEPNP_P3P

        # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
        # The default method used to estimate the camera pose for the Minimal Sample Sets step is SOLVEPNP_EPNP.
        # Exceptions are:
        # if you choose SOLVEPNP_P3P or SOLVEPNP_AP3P, these methods will be used.
        # if the number of input points is equal to 4, SOLVEPNP_P3P is used
        ret = cv2.solvePnPRansac(pts3d_curr,
                                 pts2d_near_filtered,
                                 K[:3,:3],
                                 distCoeffs=None,
                                 iterationsCount=100,
                                 reprojectionError=2.0,
                                 flags=flag)
        success = ret[0]
        rotation_vector = ret[1]
        translation_vector = ret[2]
        return (success, rotation_vector, translation_vector)
    else:
        return (False, None, None)

def matrix_to_axis_angle(R):
    """
    R: [B, 3, 3] rotation matrix
    return: [B, 3] axis-angle vector
    """
    batch_size = R.shape[0]
    cos_angle = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2
    angle = torch.acos(torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6))

    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = torch.stack([rx, ry, rz], dim=1)
    axis = axis / (2 * torch.sin(angle)).unsqueeze(1).clamp(min=1e-6)
    
    return axis * angle.unsqueeze(1)  # [B, 3]

def matrix_to_vec(T: torch.Tensor) -> torch.Tensor:
    """
    T: [B, 4, 4] SE(3) pose matrix
    return: [B, 6] (translation + axis-angle rotation)
    """
    rot = T[:, :3, :3]
    trans = T[:, :3, 3]
    rot_vec = matrix_to_axis_angle(rot)
    return torch.cat([trans, rot_vec], dim=1)

class PnPPose(nn.Module):
    """
    Canonical motion estimation module.
    """
    def __init__(self, cfg):
        super(PnPPose, self).__init__()
        self.read_config(cfg)

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def forward(self, inputs, frame_ids, _):
        outputs = {}
    
        # initialize dictionary
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        lev = self.fusion_level
        
        cam = 0 # canonical camera
        K = inputs[('K', 0)][:,cam,:,:]
        invK = inputs[('inv_K', 0)][:,cam,:,:]
        src_img = inputs[('color', frame_ids[0], 0)][0,cam,:,:]
        tar_img = inputs[('color', frame_ids[1], 0)][0,cam,:,:]
        depth = inputs[('gt_depth', 0)][0,cam,:,:]

        rgb_curr = (src_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rgb_near = (tar_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        depth_curr = depth.permute(1, 2, 0).cpu().numpy()
        K = K[0,:3,:3].cpu().numpy()

        
        gray_curr = rgb2gray(rgb_curr).astype(np.uint8)
        gray_near = rgb2gray(rgb_near).astype(np.uint8)

        pts2d_curr, pts2d_near = feature_match(gray_curr, gray_near)  # feature matching
        
        # dilation of depth
        kernel = np.ones((4, 4), np.uint8)
        depth_curr_dilated = cv2.dilate(depth_curr, kernel)

        # extract 3d pts
        pts3d_curr = []
        pts2d_near_filtered = []  # keep only feature points with depth in the current frame
        for i, pt2d in enumerate(pts2d_curr):
            # print(pt2d)
            u, v = pt2d[0], pt2d[1]
            z = depth_curr_dilated[v, u]
            if z > 0:
                xyz_curr = convert_2d_to_3d(u, v, z, K)
                pts3d_curr.append(xyz_curr)
                pts2d_near_filtered.append(pts2d_near[i])

        pts3d_curr = np.expand_dims(np.array(pts3d_curr).astype(np.float32), axis=1)
        pts2d_near_filtered = np.expand_dims(np.array(pts2d_near_filtered).astype(np.float32), axis=1)

        same_length = pts3d_curr.shape[0] == pts2d_near_filtered.shape[0]
        # the minimal number of points accepted by solvePnP is 4:
        required_count = pts3d_curr.shape[0] >= 4
        
        device = depth.device
        if same_length and required_count:
            pt3ds_torch = torch.from_numpy(pts3d_curr.squeeze(1)).to(device) # [N, 3]
            pt2ds_torch = torch.from_numpy(pts2d_near_filtered.squeeze(1)).to(device) # [N, 2]
            K_torch = torch.from_numpy(K).to(device) # [3, 3]
            num_points = pt3ds_torch.shape[0] 
            
            ### all theseus variables should have batch dimension
            #init = np.concatenate((quat_wxyz,translation_vector.reshape(1, 3)[0]),axis=0)
            init = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            T_cw = th.SE3(x_y_z_quaternion=torch.FloatTensor([init]).to(device), name="T_cw")
            # T_cw = th.SE3(tensor=TRANSFORMATION4X4_MATRIX, name="T_cw") # tensor [1, 4, 4]
            pt3ds = th.Variable(pt3ds_torch.unsqueeze(dim=0), name="obj_points") # [1, N, 3]
            pt2ds = th.Variable(pt2ds_torch.unsqueeze(dim=0), name="img_points") # [1, N, 2]
            K_th = th.Variable(K_torch.unsqueeze(dim=0), name="K_th") # [1, 3, 3]

            ### grouping in list
            optim_vars = [T_cw]
            aux_vars = [pt3ds, pt2ds, K_th]
            
            # similar to CERES AutoDiffCostFunction.
            cost_function = th.AutoDiffCostFunction(
                            optim_vars,
                            error_fn,
                            num_points,
                            aux_vars=aux_vars,
                            name="cost_fn",
                            # autograd_vectorize 기본이 False지만, autograd_mode 기본이 VMAP 이므로
                            autograd_vectorize=False,
                            #autograd_mode=AutogradMode.LOOP_BATCH
                        )
                                    
            objective = th.Objective().to(device)
            objective.add(cost_function)
            #objective.disable_vectorization()
            
            optimizer = th.LevenbergMarquardt(
                        objective,
                        max_iterations=100,
                        step_size=0.1,
                        #vectorize=False,  # 여기도 꺼주고
                    )
            th_optim = th.TheseusLayer(
                        optimizer,
                        #vectorize=False,  # 그리고 여기까지!
                    ).to(device)
            

            updated_inputs, info = th_optim.forward()

            cam_T_cam = torch.cat((updated_inputs["T_cw"], torch.tensor([[[0,0,0,1]]]).to(device)), dim=1)
            cam_T_cam = cam_T_cam.inverse()
            print(cam_T_cam)
            pose_6d = matrix_to_vec(cam_T_cam) # [trans, rot_vec]
            axis_angle, translation = pose_6d[0,3:].unsqueeze(0).unsqueeze(0), pose_6d[0,:3].unsqueeze(0).unsqueeze(0)
        
        else:
            print("Different Shape or Not Slove PnP")        
                
        return cam_T_cam, axis_angle, translation