# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv2d, pack_cam_feat, unpack_cam_feat
#from .volumetric_fusionnet import VFNet

from external.layers import ResnetEncoder, PoseDecoder

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

class GTPose(nn.Module):
    """
    Canonical motion estimation module.
    """
    def __init__(self, cfg):
        super(GTPose, self).__init__()
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
        # inputs[('pose', frame_ids[0])] -> (1,6,4,4)
        # frame_ids[1]에서 관찰된 포즈를 frame_ids[0]의 기준 좌표계로 변환한 상대 포즈
        # frame_ids[0] 좌표계에서 frame_ids[1]의 상대적 변환
        # @: 왼쪽에서부터 연산 시작
        cam_T_cam = torch.inverse(inputs[('pose', frame_ids[0])][:,0,:,:]) @ inputs[('pose', frame_ids[1])][:,0,:,:]
        pose_6d = matrix_to_vec(cam_T_cam) # [trans, rot_vec]
        axis_angle, translation = pose_6d[0,3:].unsqueeze(0).unsqueeze(0), pose_6d[0,:3].unsqueeze(0).unsqueeze(0)
        
        return cam_T_cam, axis_angle, translation