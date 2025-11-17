# Copyright (c) 2023 42dot. All rights reserved.
import torch
from pytorch3d.transforms import matrix_to_euler_angles 

from .loss_util import compute_photometric_loss, compute_masked_loss
from .single_cam_loss import SingleCamLoss

import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
#from pytorch3d.loss import chamfer_distance

import torchvision.utils as vutils
import matplotlib.pyplot as plt

def backproject(invK, depth):
    """
    This function back-projects 2D image points to 3D.
    """

    # initialize img point grid
    device = 'cuda'
    batch_size, _, height, width = depth.shape

    img_points = np.meshgrid(range(width), range(height), indexing='xy')
    img_points = torch.from_numpy(np.stack(img_points, 0)).float()
    img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
    img_points = img_points.to(device)

    to_homo = torch.ones([batch_size, 1, width*height]).to(device)
    homo_points = torch.cat([img_points, to_homo], 1)

    depth = depth.view(batch_size, 1, -1)
    points3D = torch.matmul(invK[:, :3, :3], homo_points)
    points3D = depth*points3D

    return torch.cat([points3D, to_homo], 1)

def rgbd_backproject(invK, rgbd):
    """
    This function back-projects 2D image points to 3D.
    """

    # initialize img point grid
    device = 'cuda'
    batch_size, _, height, width = rgbd.shape

    img_points = np.meshgrid(range(width), range(height), indexing='xy')
    img_points = torch.from_numpy(np.stack(img_points, 0)).float()
    img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
    img_points = img_points.to(device)

    to_homo = torch.ones([batch_size, 1, width*height]).to(device)
    homo_points = torch.cat([img_points, to_homo], 1)

    depth = rgbd[:,3,:,:]
    color = rgbd[:,:3,:,:]

    depth = depth.view(batch_size, 1, -1)
    color = color.view(batch_size, 3, -1)

    points3D = torch.matmul(invK[:, :3, :3], homo_points)
    points3D = depth*points3D

    points3D_with_color = torch.cat([points3D, color], dim=1)

    return points3D_with_color

class MultiCamLoss(SingleCamLoss):
    """
    Class for multi-camera(spatio & temporal) loss calculation
    """
    def __init__(self, cfg, rank):
        super(MultiCamLoss, self).__init__(cfg, rank)
        self.lidar_loss_list = []
        
    def compute_spatio_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None):
        """
        This function computes spatial loss.
        """
        #tar_gt_depth = inputs[('gt_depth',0)][:, cam, ...]
        #tar_gt_depth_mask = (tar_gt_depth == 0)
        
        # self occlusion mask * overlap region mask
        spatio_mask = ref_mask * target_view[('overlap_mask', 0, scale)]
        loss_args = {
            'pred': target_view[('overlap', 0, scale)],
            'target': inputs['color',0, 0][:,cam, ...]       
        }        
        spatio_loss = compute_photometric_loss(**loss_args)
        
        target_view[('overlap_mask', 0, scale)] = spatio_mask         
        return compute_masked_loss(spatio_loss, spatio_mask) 

    def compute_spatio_tempo_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in self.frame_ids[1:]:

            pred_mask = ref_mask * target_view[('overlap_mask', frame_id, scale)]
            pred_mask = pred_mask * reproj_loss_mask 
            
            loss_args = {
                'pred': target_view[('overlap', frame_id, scale)],
                'target': inputs['color',0, 0][:,cam, ...]
            } 
            
            spatio_tempo_losses.append(compute_photometric_loss(**loss_args))
            spatio_tempo_masks.append(pred_mask)
        
        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 1)
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 1)    

        # for the loss, take minimum value between reprojection loss and identity loss(moving object)
        # for the mask, take maximum value between reprojection mask and overlap mask to apply losses on all the True values of masks.
        spatio_tempo_loss, _ = torch.min(spatio_tempo_losses, dim=1, keepdim=True)
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=1, keepdim=True)
     
        return compute_masked_loss(spatio_tempo_loss, spatio_tempo_mask) 
    
    def compute_pose_con_loss(self, inputs, outputs, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes pose consistency loss in "Full surround monodepth from multiple cameras"
        """        
        ref_output = outputs[('cam', 0)]
        ref_ext = inputs['extrinsics'][:, 0, ...]
        ref_ext_inv = inputs['extrinsics_inv'][:, 0, ...]
   
        cur_output = outputs[('cam', cam)]
        cur_ext = inputs['extrinsics'][:, cam, ...]
        cur_ext_inv = inputs['extrinsics_inv'][:, cam, ...] 
        
        trans_loss = 0.
        angle_loss = 0.
     
        for frame_id in self.frame_ids[1:]:
            ref_T = ref_output[('cam_T_cam', 0, frame_id)]
            cur_T = cur_output[('cam_T_cam', 0, frame_id)]    

            # ref to world @ world to cur @ cur_T @ cur to world @ world to ref -> 
            cur_T_aligned = ref_ext_inv@cur_ext@cur_T@cur_ext_inv@ref_ext

            ref_ang = matrix_to_euler_angles(ref_T[:,:3,:3], 'XYZ')
            cur_ang = matrix_to_euler_angles(cur_T_aligned[:,:3,:3], 'XYZ')

            ang_diff = torch.norm(ref_ang - cur_ang, p=2, dim=1).mean()
            t_diff = torch.norm(ref_T[:,:3,3] - cur_T_aligned[:,:3,3], p=2, dim=1).mean()

            trans_loss += t_diff
            angle_loss += ang_diff
        
        pose_loss = (trans_loss + 10 * angle_loss) / len(self.frame_ids[1:])
        return pose_loss

    '''
    def compute_lidar_loss_3d(self, inputs, target_view, kn=3, cam=0, scale=0, ref_mask=None):
        rgb = inputs[('color', 0, 0)][:,cam,:,:,:]
        invK = inputs[('inv_K', 0)][:,cam,:,:]

        gt_depth = inputs['sparse_depth'][:,cam,:,:]
        #rgbd = torch.cat([rgb, gt_depth], dim=1)
        gt_dpts = backproject(invK, gt_depth)
        
        for scale in self.scales:
            pred_depth = target_view[('depth', scale)] #outputs[('cam', cam)][('depth',0)]
            #rgbd = torch.cat([rgb, pred_depth], dim=1)
            pred_dpts = backproject(invK, pred_depth)

            # 예측과 GT 깊이 오차 큰 100개의 인덱스 추출
            large_loss, large_loss_idx = torch.topk(gt_dpts[:,2,:] - pred_dpts[:,2,:], k=100)

            # 조건: z > 0
            mask = (gt_dpts[:,2,:] != 0)#(xyz > 0).all(dim=1)  # z가 0보다 큰 점의 마스크 (1, n)

            # 인덱스 추출
            valid_idx = torch.nonzero(mask, as_tuple=True)[1]  # n축의 인덱스만 반환
            #valid_idx = valid_idx[:100] # 연산량 많기 때문에 n개 point만 비교 (향후에 객체 부분이나 2d 상에서 depth map 차이 큰 부분만 골라서 추출)

            # valid_idx와 large_loss_idx의 교집합 구하기
            # torch.isin(elements, test_elements): elements가 test_elements에 있는지 (elements 개수가 더 많아야함)
            combined_idx = valid_idx[torch.isin(valid_idx, large_loss_idx)]

            # 디버그 출력
            #print("valid_idx:", valid_idx)
            #print("large_loss_idx:", large_loss_idx)
            #print("combined_idx:", combined_idx)
            #print(combined_idx.shape)

            loss = 0

            for idx in combined_idx:
                tar_p = gt_dpts[0,:,idx].unsqueeze(dim=1)
                _, c, n = gt_dpts.shape
                tar_p = tar_p.expand(c, n)

                # 현재 gt_dpts(tar_p)의 근접점 kn개 찾기
                dist = torch.sqrt(torch.sum((tar_p[:3,:] - gt_dpts[0,:3,:]) ** 2, dim=0))
                nn_pts, nn_idx = torch.topk(dist, k=kn+1, largest=False) # kn+1: 자기 자신이 가장 최솟값일텐데 이걸 제외한 근접점을 얻기 위해

                pred_tar_p = pred_dpts[0,:,valid_idx[0]].unsqueeze(dim=1)
                _, c, n = pred_dpts.shape
                pred_tar_p = pred_tar_p.expand(c, n)

                # 위에서 구한 gt_dpts의 인덱스 정보로 예측 포인트간 거리 최소화 하여 loss 계산
                x = pred_tar_p[:,:kn]
                y = pred_dpts[0,:,nn_idx[1:]]
                loss += torch.mean(torch.sqrt(torch.sum((x - y) ** 2, dim=0)))
            
        return torch.log(1 + torch.mean(loss))
        
    
    def compute_lidar_loss_3d(self, inputs, target_view, kn=3, cam=0, scale=0, ref_mask=None):
        rgb = inputs[('color', 0, 0)][:, cam, :, :, :]
        invK = inputs[('inv_K', 0)][:, cam, :, :]

        gt_depth = inputs['sparse_depth'][:, cam, :, :]
        gt_dpts = backproject(invK, gt_depth)
        
        for scale in self.scales:
            pred_depth = target_view[('depth', scale)]
            pred_dpts = backproject(invK, pred_depth)

            large_loss, large_loss_idx = torch.topk(gt_dpts[:, 2, :] - pred_dpts[:, 2, :], k=100)

            mask = (gt_dpts[:, 2, :] != 0)
            valid_idx = torch.nonzero(mask, as_tuple=True)[1]

            combined_idx = valid_idx[torch.isin(valid_idx, large_loss_idx)]

            if combined_idx.numel() == 0:
                continue  # Skip if no valid indices

            loss = 0
            for idx in combined_idx:
                tar_p = gt_dpts[0, :, idx].unsqueeze(dim=1)
                _, c, n = gt_dpts.shape
                tar_p = tar_p.expand(c, n)

                dist = torch.sqrt(torch.clamp(torch.sum((tar_p[:3, :] - gt_dpts[0, :3, :]) ** 2, dim=0), min=1e-6))
                nn_pts, nn_idx = torch.topk(dist, k=kn + 1, largest=False)

                pred_tar_p = pred_dpts[0, :, idx].unsqueeze(dim=1)
                pred_tar_p = pred_tar_p.expand(c, n)

                x = pred_tar_p[:, :kn]
                y = pred_dpts[0, :, nn_idx[1:]]
                loss += torch.mean(torch.sqrt(torch.clamp(torch.sum((x - y) ** 2, dim=0), min=1e-6)))
    
        return torch.log(1 + torch.clamp(torch.mean(loss), min=0))
    '''
    
    def compute_lidar_loss_3d(self, inputs, target_view, kn=3, cam=0, scale=0, ref_mask=None):
        
        rgb = inputs[('color', 0, 0)][:, cam, :, :, :]
        invK = inputs[('inv_K', 0)][:, cam, :, :]

        gt_depth = inputs[('gt_depth', 0)][:, cam, :, :]
        gt_dpts = backproject(invK, gt_depth)
        
        total_loss = 0
        for scale in self.scales:
            pred_depth = target_view[('depth', scale)]
            pred_dpts = backproject(invK, pred_depth)
            
            # 조건: z > 0
            mask = (gt_dpts[:,2,:] != 0)#(xyz > 0).all(dim=1)  # z가 0보다 큰 점의 마스크 (1, n)

            # 인덱스 추출
            valid_idx = torch.nonzero(mask, as_tuple=True)[1]  # n축의 인덱스만 반환
            gt_dpts = gt_dpts[:,:3,valid_idx]
            pred_dpts = pred_dpts[:,:3,valid_idx]
            
            loss = torch.abs(pred_dpts - gt_dpts).mean()
            total_loss += loss

        return total_loss / len(self.scales)
    
    def compute_lidar_loss_2d(self, inputs, target_view, cam=0, scale=0, ref_mask=None):

        single_loss = 0
        
        min_depth, max_depth = 0, 200
        
        for scale in self.scales:
            #print(target_view[('warp_depth', -1, scale)])
            #print(target_view[('warp_depth', 1, scale)])
            #vutils.save_image(target_view[('warp_depth', -1, scale)][0], 'wrap_cur2prev.png')
            #vutils.save_image(target_view[('warp_depth', 1, scale)][0], 'wrap_cur2next.png')
            '''
            wrap1 = target_view[('warp_depth', -1, scale)] * target_view[('warp_depth_mask', -1, scale)]
            plt.imshow(wrap1[0].permute(1, 2, 0).detach().cpu().numpy(), cmap='viridis')
            plt.savefig("wrap_cur2prev.png")
            
            wrap2 = target_view[('warp_depth', 1, scale)] * target_view[('warp_depth_mask', 1, scale)]
            plt.imshow(wrap2[0].permute(1, 2, 0).detach().cpu().numpy(), cmap='viridis')
            plt.savefig("wrap_cur2next.png")
            '''
            # single
            gt_depth = inputs[('gt_depth', 0)][:, cam, :, :]
            pred_depth = target_view[('depth', scale)]
            
            mask = (gt_depth > min_depth) * (gt_depth < max_depth) * inputs['mask'][:, cam, ...]
            mask = mask.bool()
            
            single_loss = F.l1_loss(pred_depth[mask], gt_depth[mask], reduction='mean')
            
            
        return single_loss / len(self.scales)
    
    def compute_lidar_weighted_smooth_loss(self, gt_mask, disp_map):
        """
        방법
            1. GT Depth가 있는 픽셀에 더 큰 가중치를 부여하여 스무딩 손실을 계산
            2. RGB 처럼 GT Depth도 수평, 수직 기울기 계산
               - GT Depth는 sparse 해서 수평, 수직 기울기 격차 커서 disp도 그렇게 될 수 있으므로 GT Depth에 UNet 적용? -> 잘되면 RGB Smooth 대신 사용
        
        Args:
            gt_mask (torch.Tensor): GT Depth가 있는 픽셀을 나타내는 바이너리 마스크. [B, 1, H, W]
            disp_map (torch.Tensor): 예측된 Disparity Map. [B, 1, H, W]

        Returns:
            smooth_loss (torch.Tensor): 스무딩 손실 값.
        """
        # Disparity Map의 수평 및 수직 기울기 계산
        grad_disp_x = torch.abs(disp_map[:, :, :, :-1] - disp_map[:, :, :, 1:])
        grad_disp_y = torch.abs(disp_map[:, :, :-1, :] - disp_map[:, :, 1:, :])

        # GT 마스크를 수평 및 수직 기울기에 맞게 크기 조정
        gt_mask_x = gt_mask[:, :, :, :-1]
        gt_mask_y = gt_mask[:, :, :-1, :]
        
        # GT Mask 가중치 설정
        # F = 0.1, T = 1 (T에 depth 값 or 오차 넣어보기)
        gt_weight_x = gt_mask_x.float() * 0.9 + 0.1
        gt_weight_y = gt_mask_y.float() * 0.9 + 0.1

        # 가중치 적용
        grad_disp_x *= gt_weight_x
        grad_disp_y *= gt_weight_y
        
        # 손실 계산
        smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()
        return smooth_loss

    
    def chamfer_dist_loss(self, inputs, target_view, kn=3, cam=0, scale=0, ref_mask=None):
        
        rgb = inputs[('color', 0, 0)][:, cam, :, :, :]
        invK = inputs[('inv_K', 0)][:, cam, :, :]

        gt_depth = inputs['sparse_depth'][:, cam, :, :]
        gt_dpts = backproject(invK, gt_depth)
        
        for scale in self.scales:
            pred_depth = target_view[('depth', scale)]
            pred_dpts = backproject(invK, pred_depth)
            
            # 조건: z > 0
            mask = (gt_dpts[:,2,:] != 0)#(xyz > 0).all(dim=1)  # z가 0보다 큰 점의 마스크 (1, n)

            # 인덱스 추출
            valid_idx = torch.nonzero(mask, as_tuple=True)[1]  # n축의 인덱스만 반환
            gt_dpts = gt_dpts[:,:3,valid_idx]
            pred_dpts = pred_dpts[:,:3,valid_idx]

        loss, _ = chamfer_distance(gt_dpts.permute(0, 2, 1), pred_dpts.permute(0, 2, 1))

        return loss
    
    def forward(self, inputs, outputs, cam):        
        loss_dict = {}
        cam_loss = 0. # loss across the multi-scale
        target_view = outputs[('cam', cam)]
        output_file = 'lidar_loss.csv'
        
        for scale in self.scales:
            kargs = {
                'cam': cam,
                'scale': scale,
                'ref_mask': inputs['mask'][:,cam,...]
            }
                          
            #reprojection_loss = self.compute_reproj_loss(inputs, target_view, **kargs)
            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)
            #spatio_loss = self.compute_spatio_loss(inputs, target_view, **kargs)
            lidar_single_loss, lidar_tempo_loss, lidar_sptio_loss, lidar_spatio_tempo_loss = self.compute_lidar_loss_3d(inputs, target_view, **kargs)
            print(cam,' = ', lidar_loss)
            '''
            self.lidar_loss_list.append(lidar_loss)
            if cam == 5:
                df = pd.DataFrame([self.lidar_loss_list])
    
                if not os.path.exists(output_file):
                    df.to_csv(output_file, index=False, mode='w', encoding='utf-8-sig')
                else:
                    df.to_csv(output_file, index=False, mode='a', encoding='utf-8-sig', header=False)
                
                self.lidar_loss_list = []
            '''
            kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
            spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs)   
            
            # pose consistency loss
            if self.pose_model == 'fsm' and cam != 0:
                pose_loss = self.compute_pose_con_loss(inputs, outputs, **kargs)
            else:
                pose_loss = 0
            
            cam_loss += reprojection_loss
            cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)            
            cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss                            
            cam_loss += self.pose_loss_coeff* pose_loss
            cam_loss += lidar_single_loss
            cam_loss += lidar_tempo_loss
            
            ##########################
            # for logger
            ##########################
            if scale == 0:
                loss_dict['reproj_loss'] = reprojection_loss.item()
                loss_dict['spatio_loss'] = spatio_loss.item()
                loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
                loss_dict['lidar_single_loss'] = lidar_single_loss.item()
                loss_dict['lidar_tempo_loss'] = lidar_tempo_loss.item()
                loss_dict['smooth'] = smooth_loss.item()
                if self.pose_model == 'fsm' and cam != 0:
                    loss_dict['pose'] = pose_loss.item()
                
                # log statistics
                self.get_logs(loss_dict, target_view, cam)
        
        cam_loss /= len(self.scales)
        return cam_loss, loss_dict