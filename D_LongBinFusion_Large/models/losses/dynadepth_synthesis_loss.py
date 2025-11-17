# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_util import compute_masked_loss
from .multi_cam_loss_lidar import MultiCamLoss

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

def interp(x, shape, mode='bilinear', align_corners=False):
    """ Image tensor interpolation of x with shape (B, C, H, W) -> (B, C, *shape)
    """
    return torch.nn.functional.interpolate(x, shape, mode=mode, align_corners=align_corners)

def compute_smooth_loss222(inp, img=None):
    """ Computes the smoothness loss for an arbitrary tensor of size [B, C, H, W]
        The color image is used for edge-aware smoothness
    """

    grad_inp_x = torch.abs(inp[:, :, :, :-1] - inp[:, :, :, 1:])
    grad_inp_y = torch.abs(inp[:, :, :-1, :] - inp[:, :, 1:, :])

    if img is not None:
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_inp_x *= torch.exp(-grad_img_x)
        grad_inp_y *= torch.exp(-grad_img_y)

    return grad_inp_x.mean() + grad_inp_y.mean()

def save_tensor_as_colormap(tensor, save_path, cmap='magma'):
    """
    PyTorch Tensor를 다양한 컬러맵으로 저장하는 함수
    :param tensor: (B, 1, H, W) 형태의 PyTorch Tensor
    :param save_path: 저장할 파일 경로
    :param cmap: 적용할 컬러맵 (예: 'magma', 'jet', 'viridis')
    """
    # CPU로 이동 후 NumPy 변환
    tensor = tensor.detach().cpu().squeeze(0).squeeze(0).numpy()  # (H, W) 형태로 변환

    # 컬러맵 적용 후 저장
    plt.imsave(save_path, tensor, cmap=cmap)

'''
def compute_smooth_loss222(inp, img=None):
    """ Computes the smoothness loss for an arbitrary tensor of size [B, C, H, W]
        The color image is used for edge-aware smoothness
    """
    grad_inp_x = torch.abs(inp[:, :, :, :-1] - inp[:, :, :, 1:])
    grad_inp_y = torch.abs(inp[:, :, :-1, :] - inp[:, :, 1:, :])

    if img is not None:
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

        # **🔹 크기 불일치 해결 (Padding 추가)**
        grad_img_x = F.pad(grad_img_x, (0, 1), mode='replicate')  # 오른쪽에 패딩 추가
        grad_img_y = F.pad(grad_img_y, (0, 0, 0, 1), mode='replicate')  # 아래쪽에 패딩 추가

        grad_inp_x *= torch.exp(-grad_img_x)
        grad_inp_y *= torch.exp(-grad_img_y)

    return grad_inp_x.mean() + grad_inp_y.mean()
'''



class DepthSynLoss(MultiCamLoss):
    """
    Class for depth synthesis loss calculation
    """
    def __init__(self, cfg, rank):
        super().__init__(cfg, rank)
        self.bce = nn.BCEWithLogitsLoss()
        self.prob_target_zero = {}
        self.prob_target_ones = {}
        
        batch_size = 1
        
        for scale in self.scales:
            h = 384 // (2 ** scale)
            w = 640 // (2 ** scale)
            self.prob_target_zero[scale] = torch.zeros(batch_size, 1, h, w).to('cuda')
            self.prob_target_ones[scale] = torch.ones(batch_size, 1, h, w).to('cuda')
        
        self.bool_Depth = None
        self.bool_CmpFlow = None
        self.bool_MotMask = None
        self.mask_disp_thrd = 0.03

        
    def compute_aug_losses(self, output, scale):
        """
        This function computes depth augmentation loss(consistency, smoothness).
        """
        '''
        Consistency: Augmented Depth Map (aug_depth)과 Novel View에서 Warp된 Depth Map (tform_depth) 간의 차이를 최소화하는 데 사용
        Smoothness: Disparity Map(disp_aug)의 수평 및 수직 기울기를 계산하여, 픽셀 간 변화량을 최소화 -> 깊이 값이 부드럽게 변화하도록 제약
        - 수평 기울기: 같은 행(row)에 위치한 인접한 두 픽셀 간 차이
        - 수직 기울기: 같은 열(column)에 위치한 인접한 두 픽셀 간 차이
        '''
        pred_losses = []
        pred_masks = []

        aug_depth = output[('depth', scale, 'aug')]
        tform_depth = output[('tform_depth', scale)] # wrap depth == novel view depth
        tform_mask = output[('tform_depth_mask', scale)]                

        for n_d in range(len(tform_depth)):
            tform_d = tform_depth[n_d]
            tform_m = tform_mask[n_d]
            pred_loss = (aug_depth - tform_d).abs() / (aug_depth + tform_d + 1e-8)
            pred_loss = torch.clamp(pred_loss, 0., 1.)
            pred_losses.append(pred_loss)
            pred_masks.append(tform_m)
        
        pred_losses = torch.cat(pred_losses, dim=0)
        pred_masks = torch.cat(pred_masks, dim=0)
        depth_con_loss = compute_masked_loss(pred_losses, pred_masks)
        
        disp_aug = output[('disp', scale, 'aug')]
        mean_disp_aug = disp_aug.mean(2, True).mean(3, True)
        norm_disp_aug = disp_aug / (mean_disp_aug + 1e-8)
    
        grad_disp_x = torch.abs(norm_disp_aug[:, :, :, :-1] - norm_disp_aug[:, :, :, 1:])
        grad_disp_y = torch.abs(norm_disp_aug[:, :, :-1, :] - norm_disp_aug[:, :, 1:, :])
        depth_sm_loss = grad_disp_x.mean() + grad_disp_y.mean()
        return depth_con_loss, depth_sm_loss      
    

    def forward(self, inputs, outputs, cam):        
        loss_dict = {}
        cam_loss = 0. # loss across the multi-scale
        target_view = outputs[('cam', cam)]
        for scale in self.scales:
            kargs = {
                'cam': cam,
                'scale': scale,
                'ref_mask': inputs['mask'][:,cam,...]
            }
                          
            reprojection_loss  = self.compute_reproj_loss(inputs, target_view, **kargs)
            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)

            spatio_loss, spatio_tempo_loss, depth_con_loss, depthsyn_loss, lidar_single_loss = 0, 0, 0, 0, 0
            if self.bool_Depth:
                spatio_loss = self.compute_spatio_loss(inputs, target_view, **kargs)
                #lidar_single_loss, lidar_tempo_loss, lidar_spatio_loss, lidar_spatio_tempo_loss = self.compute_lidar_loss_2d(inputs, target_view, **kargs)
                lidar_single_loss = self.compute_lidar_loss_2d(inputs, target_view, **kargs)
                kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
                spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs)

                # depth synthesis
                depth_con_loss, depth_sm_loss = self.compute_aug_losses(target_view, scale)
                depthsyn_loss = self.depth_con_coeff * depth_con_loss + self.depth_sm_coeff * depth_sm_loss
            
            cam_loss += reprojection_loss
            cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss             
            cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            cam_loss += depthsyn_loss
            cam_loss += lidar_single_loss
            #cam_loss += lidar_smooth_loss

            # pose consistency loss
            if self.pose_model == 'fsm' and cam != 0:
                pose_loss = self.compute_pose_con_loss(inputs, outputs, **kargs)
            else:
                pose_loss = 0
            

            # Motion Regularization  
            num_frames = len(self.frame_ids[1:])
            c_smooth, c_consistency, m_sparsity, m_smooth = 0, 0, 0, 0

            for frame_id in self.frame_ids[1:]:

                disp = target_view[('disp', scale)]                          # (B, 1, h, w) 
                color = inputs[('color', 0, scale)][:, cam, ...]                         # (B, 1, h, w) 
                motion_mask = target_view[('motion_mask', frame_id, scale)]     # (B, 1, h, w)
                h, w = motion_mask.shape[-2:]
                
                #tar_mask = inputs[('gt_mask', 0)][:,cam,:,:]

                #vutils.save_image(motion_mask, '/workspace/DynaVFDepth/debug_motion/motion_mask.png')
                
                if self.bool_CmpFlow:
                    complete_flow = target_view[('complete_flow', frame_id, scale)]     # (B, 3, h, w)
                    residual_flow = target_view[('residual_flow', frame_id, scale)]     # (B, 3, h, w)

                    #vutils.save_image(complete_flow, '/workspace/DynaVFDepth/debug_motion/complete_flow.png')
                    #vutils.save_image(residual_flow, '/workspace/DynaVFDepth/debug_motion/residual_flow.png')

                    #if losses['loss_coef/c_smooth'] > 0:
                    c_smooth += compute_smooth_loss222(complete_flow, color) / (2 ** scale) 

                    # consistency can only be computed when the motion mask is predicted as well
                    if self.bool_MotMask: #and losses['loss_coef/c_consistency'] > 0:
                        valid_disp = (disp > self.mask_disp_thrd).detach()  # avoid rotational edge cases
                        c_consistency += torch.mean(valid_disp * (1-motion_mask.detach()) * torch.abs(residual_flow)) / (2 ** scale)
                
                if self.bool_MotMask:
                    sample_ego = target_view[('sample_ego', frame_id, scale)]               # (B, H, W, 2) 
                    sample_complete = target_view[('sample_complete', frame_id, scale)]     # (B, H, W, 2)
                    motion_prob = target_view[('motion_prob', frame_id, scale)]             # (B, 1, h, w)
                
                    #vutils.save_image(motion_prob, '/workspace/DynaVFDepth/debug_motion/motion_prob.png')
                
                

                    #if losses['loss_coef/m_sparsity'] > 0:
                    sample_ego = interp(sample_ego.permute(0, 3, 1, 2), (h, w))             # (B, 2, h, w) 
                    sample_complete = interp(sample_complete.permute(0, 3, 1, 2), (h, w))   # (B, 2, h, w) 
                    disp_mag = torch.sum((sample_ego - sample_complete) ** 2, 1)            # (B, h, w) 
                    static = (disp_mag < disp_mag.mean()).unsqueeze(1)                      # (B, 1, h, w) 
                    dynamic = (disp_mag >= disp_mag.mean()).unsqueeze(1)
                    
                    if torch.all(torch.sum(static, (1,2,3)) > 0):
                        m_sparsity += self.bce(motion_prob[static], self.prob_target_zero[scale][static]) / (2 ** scale) 
                        #m_sparsity += self.bce(motion_prob[dynamic], self.prob_target_ones[scale][dynamic]) / (2 ** scale) 
                    
                    
                    #if losses['loss_coef/m_smooth'] > 0:
                    m_smooth += compute_smooth_loss222(motion_mask, color) / (2 ** scale) 
                    
                    #static = 1.0 - tar_mask
                    #static_flow_loss = F.l1_loss(sample_ego*static, sample_complete*static, reduction='mean')
                    #motion_mask_loss = self.bce(motion_mask, tar_mask) 
            
            #cam_loss += (static_flow_loss / num_frames)
            #cam_loss += (motion_mask_loss / num_frames)
                    
                    
                    #if torch.all(torch.sum(static, (1,2,3)) > 0):
                    #    m_sparsity += self.bce(motion_prob[static], self.prob_target_zero[scale][static]) / (2 ** scale) / num_frames
                    #    m_sparsity += self.bce(motion_prob[dynamic], self.prob_target_ones[scale][dynamic]) / (2 ** scale) / num_frames
                    
                    #if losses['loss_coef/m_smooth'] > 0:
                    #m_smooth += compute_smooth_loss222(motion_mask, color) / (2 ** scale) / num_frames

                #save_tensor_as_colormap(motion_mask, '/workspace/DynaVFDepth/debug_motion/motion_mask.png', cmap='jet')
                #save_tensor_as_colormap(complete_flow.norm(dim=1, keepdim=True), '/workspace/DynaVFDepth/debug_motion/complete_flow.png', cmap='jet')
                #save_tensor_as_colormap(residual_flow.norm(dim=1, keepdim=True), '/workspace/DynaVFDepth/debug_motion/residual_flow.png', cmap='jet')
                #save_tensor_as_colormap(motion_prob, '/workspace/DynaVFDepth/debug_motion/motion_prob.png', cmap='jet')

            cam_loss += (c_smooth / num_frames)
            cam_loss += (c_consistency / num_frames)
            cam_loss += (m_sparsity / num_frames)
            cam_loss += (m_smooth / num_frames)
            

            ##########################
            # for logger
            ##########################
            if scale == 0:
                loss_dict['tempo_loss'] = reprojection_loss.item()
                loss_dict['smooth'] = smooth_loss.item()
                if self.bool_Depth:
                    loss_dict['spatio_loss'] = spatio_loss.item()
                    loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
                    loss_dict['depth_loss'] = depthsyn_loss.item()
                    loss_dict['depth_sm_loss'] = depth_sm_loss.item()
                    loss_dict['depth_con_loss'] = depth_con_loss.item()                    
                    loss_dict['lidar_single_loss'] = lidar_single_loss.item()
                    #loss_dict['static_flow_loss'] = static_flow_loss.item()
                    #loss_dict['motion_mask_loss'] = motion_mask_loss.item()
                
                if self.bool_CmpFlow:
                    loss_dict['c_smooth'] = c_smooth.item()
                    loss_dict['c_consistency'] = c_consistency.item()
                
                if self.bool_MotMask:
                    loss_dict['m_sparsity'] = m_sparsity.item()
                    loss_dict['m_smooth'] = m_smooth.item()
                
                if self.pose_model == 'fsm' and cam != 0:
                    loss_dict['pose'] = pose_loss.item()

                # log statistics
                self.get_logs(loss_dict, target_view, cam)                       
        
        cam_loss /= len(self.scales)
        loss_dict['cam_loss'] = cam_loss.item()
        return cam_loss, loss_dict

    '''
    def forward(self, inputs, outputs, cam):        
        loss_dict = {}
        cam_loss = 0. # loss across the multi-scale
        target_view = outputs[('cam', cam)]
        for scale in self.scales:
            kargs = {
                'cam': cam,
                'scale': scale,
                'ref_mask': inputs['mask'][:,cam,...]
            }
                          
            #reprojection_loss  = self.compute_reproj_loss(inputs, target_view, **kargs)
            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)
            #spatio_loss = self.compute_spatio_loss(inputs, target_view, **kargs)
            lidar_single_loss, lidar_tempo_loss, lidar_spatio_loss, lidar_spatio_tempo_loss = self.compute_lidar_loss_2d(inputs, target_view, **kargs)
            #lidar_single_loss, lidar_smooth_loss, _ = self.compute_lidar_loss_2d(inputs, target_view, **kargs)
            #print(cam,' = ', lidar_loss)
            
            #kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
            #spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs)
            
            # depth synthesis
            depth_con_loss, depth_sm_loss = self.compute_aug_losses(target_view, scale)
            depthsyn_loss = self.depth_con_coeff * depth_con_loss + self.depth_sm_coeff * depth_sm_loss
            
            cam_loss += lidar_tempo_loss #reprojection_loss
            cam_loss += self.spatio_coeff * lidar_spatio_loss + self.spatio_tempo_coeff * lidar_spatio_tempo_loss             
            cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            cam_loss += depthsyn_loss
            cam_loss += lidar_single_loss
            #cam_loss += lidar_smooth_loss
            
            ##########################
            # for logger
            ##########################
            if scale == 0:
                loss_dict['tempo_loss'] = lidar_tempo_loss.item() #reprojection_loss.item()
                loss_dict['spatio_loss'] = lidar_spatio_loss.item()
                loss_dict['spatio_tempo_loss'] = lidar_spatio_tempo_loss.item()
                loss_dict['depth_loss'] = depthsyn_loss.item()
                loss_dict['depth_sm_loss'] = depth_sm_loss.item()
                loss_dict['depth_con_loss'] = depth_con_loss.item()                    
                loss_dict['smooth'] = smooth_loss.item()
                loss_dict['lidar_single_loss'] = lidar_single_loss.item()
                #loss_dict['lidar_smooth_loss'] = lidar_smooth_loss.item()

                # log statistics
                self.get_logs(loss_dict, target_view, cam)                       
        
        cam_loss /= len(self.scales)
        loss_dict['cam_loss'] = cam_loss.item()
        return cam_loss, loss_dict
    '''