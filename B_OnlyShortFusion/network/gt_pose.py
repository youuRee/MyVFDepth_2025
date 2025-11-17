# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv2d, pack_cam_feat, unpack_cam_feat
#from .volumetric_fusionnet import VFNet

from external.layers import ResnetEncoder, PoseDecoder

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
        rel_pose = torch.inverse(inputs[('pose', frame_ids[0])][:,0,:,:]) @ inputs[('pose', frame_ids[1])][:,0,:,:]
        
        return rel_pose