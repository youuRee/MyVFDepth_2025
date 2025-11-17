# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv2d, pack_cam_feat, unpack_cam_feat
#from .volumetric_fusionnet import VFNet

from external.layers import ResnetEncoder, PoseDecoder


def se3_inv(T):
    R = T[..., :3, :3]
    t = T[..., :3,  3]
    RT = R.transpose(-1, -2)
    Ti = torch.eye(4, device=T.device, dtype=T.dtype).expand_as(T).clone()
    Ti[..., :3, :3] = RT
    Ti[..., :3,  3] = -(RT @ t.unsqueeze(-1)).squeeze(-1)
    return Ti

def relative(T_a_to_w_now, T_a_to_w_next):
    # 같은 좌표계 a의 포즈 두 개에서 "now→next" 상대변환
    return se3_inv(T_a_to_w_now) @ T_a_to_w_next

def rot_angle_deg(R):
    # 수치 안정화
    tr = R[..., 0,0] + R[..., 1,1] + R[..., 2,2]
    cos_theta = (tr - 1.) / 2.
    cos_theta = torch.clamp(cos_theta, -1., 1.)
    return torch.rad2deg(torch.acos(cos_theta))

def pose_error(T_rel_ref, T_rel_est):
    # 회전/병진 오차 (ref: 위, est: 아래)
    R_ref, t_ref = T_rel_ref[..., :3, :3], T_rel_ref[..., :3, 3]
    R_est, t_est = T_rel_est[..., :3, :3], T_rel_est[..., :3, 3]
    R_err = R_est @ R_ref.transpose(-1, -2)
    ang_err = rot_angle_deg(R_err)
    trans_err = torch.linalg.norm(t_est - t_ref, dim=-1)
    return ang_err, trans_err


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

class LidarSlamPose(nn.Module):
    """
    Canonical motion estimation module.
    """
    def __init__(self, cfg):
        super(LidarSlamPose, self).__init__()
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
        #gt_cam_T_cam = torch.inverse(inputs[('pose', frame_ids[0])][:,0,:,:]) @ inputs[('pose', frame_ids[1])][:,0,:,:] # [1,4,4]
        #print('frame: ', frame_ids[0], ' and ', frame_ids[1])
        #print('Camera Rel Pose:\n', gt_cam_T_cam)
        
        # frame_ids = [-1, 0], [0, 1]
        lidar_pose_0 = inputs[('L_pose', frame_ids[0])][:,0,:,:]
        lidar_pose_1 = inputs[('L_pose', frame_ids[1])][:,0,:,:]
        
        cam_extr = inputs['extrinsics'][:,0,:,:] # cam0 기준
        lidar_to_cam_0 = lidar_pose_0 @ cam_extr
        lidar_to_cam_1 = lidar_pose_1 @ cam_extr
        cam_T_cam = torch.inverse(lidar_to_cam_0) @ lidar_to_cam_1 # [1,4,4]
        
        #print('LiDAR to Camera Rel Pose:\n', cam_T_cam)
        #ang_err_deg, trans_err_m = pose_error(gt_cam_T_cam, cam_T_cam)
        #print("\nRotation error (deg):", ang_err_deg.item())
        #print("Translation error (m):", trans_err_m.item())
        #print('--------------------------------------------------------------------')
        
        pose_6d = matrix_to_vec(cam_T_cam) # [trans, rot_vec]
        axis_angle, translation = pose_6d[0,3:].unsqueeze(0).unsqueeze(0), pose_6d[0,:3].unsqueeze(0).unsqueeze(0)
        
        return cam_T_cam, axis_angle, translation