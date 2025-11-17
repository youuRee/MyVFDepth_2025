# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_util import Projection
from .dynamic_util import BackprojectDepth, Project3D

import numpy as np
import matplotlib.pyplot as plt
import cv2

def interp(x, shape, mode='bilinear', align_corners=False):
    """ Image tensor interpolation of x with shape (B, C, H, W) -> (B, C, *shape)
    """
    return torch.nn.functional.interpolate(x, shape, mode=mode, align_corners=align_corners)

def flow_to_color(flow, max_flow=None):
    """
    Convert flow to color map for visualization.
    flow: (H, W, 2) numpy
    """
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    rad = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx)

    if max_flow is None:
        max_flow = np.percentile(rad, 99)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) * (180 / np.pi / 2)).astype(np.uint8)  # Hue
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(rad / max_flow, 0, 1) * 255).astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

# 과거 시점(from_pose)에 있던 점들을 현재 시점(to_pose)의 좌표계로 변환
def transform_point_cloud(past_point_clouds, from_pose, to_pose):
    #transformation = torch.inverse(to_pose) @ from_pose #torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
    NP = past_point_clouds.shape[0]
    xyz = torch.hstack([past_point_clouds, torch.ones(NP, 1, device=past_point_clouds.device)]).T
    world_pts = from_pose @ xyz
    past_point_clouds = (torch.inverse(to_pose) @ world_pts).T[:, :3]
    #past_point_clouds = (transformation @ xyz1).T[:, :3]
    return past_point_clouds

class ViewRendering(nn.Module):
    """
    Class for rendering images from given camera parameters and pixel wise depth information
    이 클래스에서 픽셀 당 뎁스 정보를 구성하므로 라이다 프로젝션 포인트 당 뎁스 정보도 여기를 참고하여 구성해보기
    """
    def __init__(self, cfg, rank):
        super().__init__()
        self.read_config(cfg)
        self.rank = rank
        self.project = self.init_project_imgs(rank)

        # Dynamo-Depth
        #self.resize = {}
        self.backproject_depth = {}
        self.project_3d = {}
        #self.prob_target = {}

        self.B = self.batch_size
        self.H = self.height
        self.W = self.width
        self.bool_CmpFlow = None
        self.bool_MotMask = None

        for scale in self.scales:
            h = self.H // (2 ** scale)
            w = self.W // (2 ** scale)

            #self.resize[scale] = torchvision.transforms.Resize((h,w),interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
            self.backproject_depth[scale] = BackprojectDepth(self.B, h, w).to('cuda')
            self.project_3d[scale] = Project3D(self.B, h, w).to('cuda')
            #self.prob_target[scale] = torch.zeros(self.B, 1, h, w).to(self.device)  
            
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_project_imgs(self, rank):
        project_imgs = {}
        project_imgs = Projection(
                self.batch_size, self.height, self.width, rank)
        return project_imgs    
    
    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features. 
        """
        _, c, h, w = mask.size()
        mean = (feature * mask).sum(dim=(1,2,3), keepdim=True) / (mask.sum(dim=(1,2,3), keepdim=True) + 1e-8)
        var = ((feature - mean) ** 2).sum(dim=(1,2,3), keepdim=True) / (c*h*w)
        return mean, torch.sqrt(var + 1e-16)     
    
    def get_norm_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()

        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1,3,1,1)

            mask_sum = mask.sum(dim=(-3,-2,-1))
            # skip when there is no overlap
            if torch.any(mask_sum == 0):
                return warp_img

            s_mean, s_std = self.get_mean_std(src_img, mask)
            w_mean, w_std = self.get_mean_std(warp_img, mask)

        norm_warp = (warp_img - w_mean) / (w_std + 1e-8) * s_std + s_mean
        return norm_warp * warp_mask.float()   

    # inverse warping src -> tar: tar 시점의 3D 정보를 바탕으로 src 이미지에서 대응하는 위치를 찾아 샘플링
    def get_virtual_image(self, src_img, src_mask, tar_depth, tar_invK, src_K, T, scale=0):
        """
        This function warps source image to target image using backprojection and reprojection process. 
        """
        # do reconstruction for target from source   
        pix_coords = self.project(tar_depth, T, tar_invK, src_K) # target depth(3D)를 src image(2D)로
        
        # 이미지를 뎁스맵에 와핑..?
        img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear', 
                                    padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest', 
                                    padding_mode='zeros', align_corners=True)
        #print("src img: ", src_img.shape) -> torch.Size([1, 3, 384, 640])
        #print("pix coords: ", pix_coords.shape) -> torch.Size([1, 384, 640, 2])
        # nan handling
        inf_img_regions = torch.isnan(img_warped)
        img_warped[inf_img_regions] = 2.0
        inf_mask_regions = torch.isnan(mask_warped)
        mask_warped[inf_mask_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, 
                                        pix_coords < -1).sum(dim=1, keepdim=True) > 0
        return img_warped, (~invalid_mask).float() * mask_warped

    def get_virtual_sparse_depth(self, src_depth, src_mask, tar_depth, tar_invK, src_K, T, min_depth, max_depth, scale=0):
    
        # do reconstruction for target from source   
        pix_coords = self.project(tar_depth, T, tar_invK, src_K) # target depth(3D)를 src image(2D)로

        # 이미지를 뎁스맵에 와핑..?
        depth_warped = F.grid_sample(src_depth, pix_coords, mode='bilinear', 
                                    padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest', 
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_depth = torch.isnan(depth_warped)
        depth_warped[inf_depth] = 2.0
        inf_regions = torch.isnan(mask_warped)
        mask_warped[inf_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, pix_coords < -1).sum(dim=1, keepdim=True) > 0

        # range handling
        valid_depth_min = (depth_warped > min_depth)
        depth_warped[~valid_depth_min] = min_depth
        valid_depth_max = (depth_warped < max_depth)
        depth_warped[~valid_depth_max] = max_depth

        return depth_warped, (~invalid_mask).float() * mask_warped * valid_depth_min * valid_depth_max
    
    def get_virtual_depth(self, src_depth, src_mask, src_invK, src_K, tar_depth, tar_invK, tar_K, T, min_depth, max_depth, scale=0):
        """
        This function backward-warp source depth into the target coordinate.
        src -> target
        """       
        # transform source depth
        b, _, h, w = src_depth.size()    
        src_points = self.project.backproject(src_invK, src_depth) # src depth로 point 만들고
        src_points_warped = torch.matmul(T[:, :3, :], src_points) # point를 tar 좌표계로 옮기고
        src_depth_warped = src_points_warped.reshape(b, 3, h, w)[:, 2:3, :, :]

        # reconstruct depth: backward-warp source depth to the target coordinate
        # 타겟 깊이 값과 T.inv를 사용하여 소스 좌표계에서 타겟 픽셀 좌표를 계산 -> F.grid_sample 시 정확한 매핑을 위해 타겟 깊이 필요
        # T.inv: 타겟 좌표계를 소스 좌표계로 변환
        # T: src -> tar => lidar temporal: src = (depth, 0), tar = (gt_depth,(-1/1))
        # T는 t -> t-1, t -> t+1 이어야 함!
        # [('cam_T_cam', 0, -1)]: t -> t-1, [('cam_T_cam', 0, 1)]: t+1 -> t
        pix_coords = self.project(tar_depth, torch.inverse(T), tar_invK, src_K)
        depth_warped = F.grid_sample(src_depth_warped, pix_coords, mode='bilinear', 
                                        padding_mode='zeros', align_corners=True) # src에서 tar로 옮긴 point를 tar pixel 좌표에 배치
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest',
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_depth = torch.isnan(depth_warped)
        depth_warped[inf_depth] = 2.0
        inf_regions = torch.isnan(mask_warped)
        mask_warped[inf_regions] = 0
        
        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, pix_coords < -1).sum(dim=1, keepdim=True) > 0

        # range handling
        valid_depth_min = (depth_warped > min_depth)
        depth_warped[~valid_depth_min] = min_depth
        valid_depth_max = (depth_warped < max_depth)
        depth_warped[~valid_depth_max] = max_depth
        return depth_warped, (~invalid_mask).float() * mask_warped * valid_depth_min * valid_depth_max        


    def forward_splat_depth_with_flow(self, depth, flow, height, width, eps=1e-6):
        """
        Forward splat depth(t) to t′ using flow(t → t′)

        Args:
            depth: (B, 1, H, W) - depth map at time t
            flow: (B, 2, H, W) - flow map from t to t′ (in pixels)
            height, width: target image size (same as input)
        Returns:
            projected_depth: (B, 1, H, W) - depth map at time t′ (z-buffered)
        """
        B, _, H, W = depth.shape
        device = depth.device

        # 1. pixel coordinate grid at time t
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=device, dtype=torch.float32),
            torch.arange(0, W, device=device, dtype=torch.float32),
            indexing='ij'
        )  # (H, W)
        pixel_coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)

        # 2. apply flow to pixel coordinates → projected pixel locations at t′
        projected_coords = pixel_coords + flow  # (B, 2, H, W)
        u_proj = projected_coords[:, 0].view(B, -1)  # (B, H*W)
        v_proj = projected_coords[:, 1].view(B, -1)

        # 3. flatten depth and projected pixel coordinates
        z_vals = depth.view(B, -1)  # (B, H*W)
        u_proj = u_proj.round().long()
        v_proj = v_proj.round().long()

        # 4. mask: valid pixel range
        valid = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)

        # 5. initialize z-buffer (inf)
        projected_depth = torch.full((B, 1, H, W), float('inf'), device=device)

        # 6. z-buffer update per batch
        for b in range(B):
            u = u_proj[b][valid[b]]
            v = v_proj[b][valid[b]]
            z = z_vals[b][valid[b]]
            idx = v * W + u  # flat indices
            proj_flat = projected_depth[b, 0].view(-1)

            # Keep nearest depth (z-buffer)
            proj_flat.index_put_((idx,), z, accumulate=False)
            projected_depth[b, 0] = proj_flat.view(H, W)

        projected_depth[projected_depth == float('inf')] = 0
        return projected_depth

    def pca_to_depth(self, accum_pts, curr_depth, H, W, k, p_cw, T_L_to_W):
        
        N = accum_pts.shape[0]
        ones = torch.ones((N, 1), device=accum_pts.device)
        X_L_homo = torch.cat([accum_pts[:, :3], ones], dim=1)
        X_W = (T_L_to_W @ X_L_homo.T).T
        Xc = (p_cw @ X_W.T).T
        cam_coords = Xc.T
        pix_coords = k @ cam_coords
        pix_coords = pix_coords[:2, :] / (pix_coords[2:3, :] + 1e-6)
        pix_coords = pix_coords.T
        depths = Xc[:, 2]

        pix_coords = pix_coords.long()
        pix_x, pix_y = pix_coords[:, 0], pix_coords[:, 1]
        valid = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H) & (depths > 0)
        pix_x = pix_x[valid]
        pix_y = pix_y[valid]
        depths = depths[valid]

        # 빈 곳만 채우기
        depth_map = curr_depth[0, 0].clone()  # 깊은 복사 (deep copy)
        #print('Before : ', (depth_map != 0).sum())
        empty_mask = depth_map[pix_y, pix_x] == 0
        depth_map[pix_y[empty_mask], pix_x[empty_mask]] = depths[empty_mask]
        #print('After : ', (depth_map != 0).sum())
        return depth_map
    
    def get_lidar_aug_depth(self, curr_pts, H, W, k, T_C_from_L_aug):
        N = curr_pts.shape[0]
        ones = torch.ones((N, 1), device=curr_pts.device)
        X_L_homo = torch.cat([curr_pts[:, :3], ones], dim=1) # 동차좌표 추가 (N, 4)

        # Camera 좌표계로 변환 (augmented extrinsics 사용!)
        #T_C_from_L_aug = inputs['extrinsics_aug'][:,cam,:,:][0].inverse()
        X_C_aug = T_C_from_L_aug @ X_L_homo.T  # [4, N]
        #k = inputs[('K', 0)][:,cam,:,:][0]

        # 이미지 평면 투영 (N, 2)
        cam_coords = X_C_aug  # (3, N)
        pix_coords = k @ cam_coords  # (3, N)
        pix_coords = pix_coords[:2, :] / pix_coords[2:3, :]  # (2, N)
        # pix_coords → (N, 2)
        pix_coords = pix_coords.T  # (N, 2), (x, y)

        depths = X_C_aug[2, :]  # 카메라 z축 값 → shape: (N,)
        pix_coords = pix_coords.long()
        pix_x, pix_y = pix_coords[:, 0], pix_coords[:, 1]

        # 이미지 범위 필터링
        valid = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H) & (depths > 0)
        pix_x = pix_x[valid]
        pix_y = pix_y[valid]
        depths = depths[valid]

        # Depth map 생성
        aug_depth_map = torch.zeros((H, W), device=depths.device)
        aug_depth_map[pix_y, pix_x] = depths  # 혹은 min() 등으로 보완
        return aug_depth_map
        
    def forward(self, inputs, outputs, cam, rel_pose_dict):
        # predict images for each scale(default = scale 0 only)
        source_scale = 0
        
        # ref inputs
        ref_color = inputs['color', 0, source_scale][:,cam, ...]        
        ref_mask = inputs['mask'][:, cam, ...]
        ref_K = inputs[('K', source_scale)][:,cam, ...]
        ref_invK = inputs[('inv_K', source_scale)][:,cam, ...]  
        
        # output
        target_view = outputs[('cam', cam)]
        
        for scale in self.scales:           
            ref_depth = target_view[('depth', scale)]
            #ref_gt_depth = inputs[('gt_depth', 0)][:,cam, ...]
            #ref_gt_depth_mask = (ref_gt_depth == 0) # 0인 부분만 True -> gt depth가 있는 픽셀은 RGB loss 반영 X
            for frame_id in self.frame_ids[1:]: # frame_ids: [0, -1, 1]

                '''
                fraim_id = -1 => T_(t-1 -> t), fraim_id = 1 => T_(t -> t+1)인데 왜 같은 T를 적용해서 와핑??
                - 실제로 차가 움직인 "절대적인 방향"보다, t 시점의 뎁스 맵을 기준으로 한 "일관된 방향"이 더 중요
                - t 시점인 뎁스 맵이 기준인 상황에서 포즈 행렬의 방향이 일관되지 않으면, 좌표계가 꼬여서 올바르게 와핑이 안됨!
                - t->t+1이 5만큼 움직였으면 t-1->t pose가 3만큼 움직여도 t->t-1 = -3 으로 바꿔서 적용하지 말고 일관성 있게 3으로 해야됨 
                '''
                # for temporal learning
                T = target_view[('cam_T_cam', 0, frame_id)]
                src_color = inputs['color', frame_id, source_scale][:, cam, ...] 
                src_mask = inputs['mask'][:, cam, ...]
                warped_img, warped_mask = self.get_virtual_image(
                    src_color, 
                    src_mask, 
                    ref_depth, 
                    ref_invK, 
                    ref_K, 
                    T, 
                    source_scale
                )
                
                
                if self.intensity_align:
                    warped_img = self.get_norm_image_single(
                        ref_color, 
                        ref_mask,
                        warped_img, 
                        warped_mask
                    )
                
                target_view[('color', frame_id, scale)] = warped_img
                target_view[('color_mask', frame_id, scale)] = warped_mask

                
            # spatio-temporal learning
            if self.spatio or self.spatio_temporal:
                for frame_id in self.frame_ids:
                    '''
                    if frame_id == 0:
                        ref_depth = (target_view[('depth', scale, 0, -1)] + target_view[('depth', scale, 0, 1)]) / 2
                    else:
                        ref_depth = target_view[('depth', scale, 0, frame_id)]
                    '''
                    overlap_img = torch.zeros_like(ref_color)
                    overlap_mask = torch.zeros_like(ref_mask)
                    #overlap_depth = torch.zeros_like(ref_gt_depth)
                    #overlap_depth_mask = torch.zeros_like(ref_mask)
                    
                    for cur_index in self.rel_cam_list[cam]:
                        # for partial surround view training
                        if cur_index >= self.num_cams: 
                            continue

                        src_color = inputs['color', frame_id, source_scale][:, cur_index, ...]
                        src_mask = inputs['mask'][:, cur_index, ...]
                        src_K = inputs[('K', source_scale)][:, cur_index, ...]                        
                        
                        rel_pose = rel_pose_dict[(frame_id, cur_index)]
                        warped_img, warped_mask = self.get_virtual_image(
                            src_color, 
                            src_mask, 
                            ref_depth, 
                            ref_invK, 
                            src_K,
                            rel_pose, 
                            source_scale
                        )

                        if self.intensity_align:
                            warped_img = self.get_norm_image_single(
                                ref_color, 
                                ref_mask,
                                warped_img, 
                                warped_mask
                            )

                        # assuming no overlap between warped images
                        overlap_img = overlap_img + warped_img
                        overlap_mask = overlap_mask + warped_mask
                        '''
                        # For LiDAR Spatio Loss
                        if frame_id == 0:
                            src_ext = inputs['extrinsics'][:, cur_index, ...]                        
                            src_depth = outputs[('cam', cur_index)][('depth', scale)]
                            #src_mask = inputs['mask'][:, cur_index, ...]                
                            src_invK = inputs[('inv_K', source_scale)][:,cur_index, ...]
                            #src_K = inputs[('K', source_scale)][:,cur_index, ...]

                            # current view to the novel view
                            warp_depth, warp_mask = self.get_virtual_depth(
                                src_depth, 
                                src_mask, 
                                src_invK, 
                                src_K,
                                ref_depth, 
                                ref_invK, 
                                ref_K, 
                                rel_pose,
                                self.min_depth,
                                self.max_depth
                            )
                            
                            target_view[('spatio_wrap_depth', cur_index, scale)] = warp_depth
                            target_view[('spatio_wrap_mask', cur_index, scale)] = warp_mask
                        '''    

                    target_view[('overlap', frame_id, scale)] = overlap_img
                    target_view[('overlap_mask', frame_id, scale)] = overlap_mask
                    
            # depth augmentation at a novel view
            if self.aug_depth:
                tform_depth = []
                tform_mask = []

                aug_ext = inputs['extrinsics_aug'][:, cam, ...]
                aug_ext_inv = torch.inverse(aug_ext)                
                aug_K, aug_invK = ref_K, ref_invK
                aug_depth = target_view[('depth', scale, 'aug')]
                
                '''
                        3 1
            rear cam <- 5   0 -> front cam
                        4 2
                        
                '''
                
                for i, curr_index in enumerate(self.rel_cam_list[cam] + [cam]):
                    # for partial surround view training
                    if curr_index >= self.num_cams: 
                        continue

                    src_ext = inputs['extrinsics'][:, curr_index, ...]                        
                    
                    src_depth = outputs[('cam', curr_index)][('depth', scale)]
                    src_mask = inputs['mask'][:, curr_index, ...]                
                    src_invK = inputs[('inv_K', source_scale)][:,curr_index, ...]
                    src_K = inputs[('K', source_scale)][:,curr_index, ...]

                    # current view to the novel view
                    rel_pose = torch.matmul(aug_ext_inv, src_ext)
                    warp_depth, warp_mask = self.get_virtual_depth(
                        src_depth, 
                        src_mask, 
                        src_invK, 
                        src_K,
                        aug_depth, 
                        aug_invK, 
                        aug_K, 
                        rel_pose,
                        self.min_depth,
                        self.max_depth
                    )
                    # aug depth에 대한 lidar supervion 필요할듯?
                    tform_depth.append(warp_depth)
                    tform_mask.append(warp_mask)

                target_view[('tform_depth', scale)] = tform_depth
                target_view[('tform_depth_mask', scale)] = tform_mask
            
            # -1, 0, 1 포인트 클라우드 누적 depth map 생성
            # [1, N, 3]
            curr_pts = inputs[('lidar', 0)]
            #past_pts = inputs[('lidar', -1)]
            #next_pts = inputs[('lidar', 1)]

            to_pose = inputs[('L_pose', 0)][0][0]
            #past_pose = inputs[('L_pose', -1)][0][0]
            #next_pose = inputs[('L_pose', 1)][0][0]
            
            #past2curr_pts = transform_point_cloud(past_pts[0], past_pose, to_pose) # [N, 3]
            #next2curr_pts = transform_point_cloud(next_pts[0], next_pose, to_pose)
            #accum_point_clouds = [past2curr_pts, next2curr_pts]
            #accum_point_clouds = torch.cat(accum_point_clouds)
            
            curr_depth = inputs[('gt_depth', 0)][:, cam, ...]
            H, W = 384, 640
            k = inputs[('K', 0)][:, cam, :, :][0]
            #p_wc = inputs[("pose", 0)][:, cam, :, :][0]
            #p_cw = p_wc.inverse()
            #T_L_to_W = inputs[('L_pose', 0)][0][0]
            
            #accum_depth = self.pca_to_depth(accum_point_clouds, curr_depth, H, W, k, p_cw, T_L_to_W)
            #target_view[('accum_depth', scale)] = accum_depth.unsqueeze(0).unsqueeze(0)
            
            T_C_from_L_aug = inputs['extrinsics_aug'][:,cam,:,:][0].inverse()
            aug_lidar_dpeth = self.get_lidar_aug_depth(curr_pts[0], H, W, k, T_C_from_L_aug)
            target_view[('aug_lidar_depth', scale)] = aug_lidar_dpeth.unsqueeze(0).unsqueeze(0)
            
            
            '''
            # SfFlow
            #if self.bool_CmpFlow:
            source_scale = 0
            for i, frame_id in enumerate(self.frame_ids[1:]):
                
                K = inputs[('K', source_scale)][:, cam, ...]
                T = target_view[('cam_T_cam', 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](ref_depth, inputs[('inv_K', source_scale)][:, cam, ...])
                target_view[('cam_points', 0, scale)] = cam_points
                
                _, _, ego_flow = self.project_3d[source_scale](cam_points, K, T) # (B, H, W, 2), (B, 3, H*W)
                ego_flow = ego_flow.permute(0, 3, 1, 2)  # (B, 2, H, W)
                
                target_view[('warp_depth', frame_id, scale)] = self.forward_splat_depth_with_flow(
                            depth=ref_depth,  # (B,1,H,W)
                            flow=ego_flow,   # (B,2,H,W)
                            height=384,
                            width=640)
                
            
                target_view[('sample_ego', frame_id, scale)] = sample_ego
                target_view[('ego_flow', frame_id, scale)] = ego_flow
                
                pred_flow = target_view[('dense_flow', frame_id, scale)]
                 (1) dense_flow: (B, 2, H, W) → (B, H, W, 2)
                flow_grid = pred_flow.permute(0, 2, 3, 1)
                valid_mask = (flow_grid != 0).float()

                flow_grid = flow_grid * valid_mask + sample_ego * (1 - valid_mask)
                target_view[('dense_flow', frame_id, scale)] = flow_grid

                
                flow_map_np = flow_grid[0].detach().cpu().numpy()
                flow_vis = flow_to_color(flow_map_np)

                plt.figure(figsize=(20, 10))
                plt.imshow(flow_vis)
                plt.title("Flow")
                plt.axis('off')
                plt.savefig('dense_flow_map.png')
                

                # (2) 정규화된 pixel grid 생성
                B, _, H, W = pred_flow.shape
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=pred_flow.device),
                    torch.linspace(-1, 1, W, device=pred_flow.device),
                    indexing='ij'
                )
                base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

                # (3) flow는 정규화된 좌표 기준이 아니므로 정규화
                norm_flow = torch.zeros_like(flow_grid)
                norm_flow[..., 0] = flow_grid[..., 0] / (W / 2)
                norm_flow[..., 1] = flow_grid[..., 1] / (H / 2)

                # (4) 최종 샘플링 위치
                sample = base_grid + norm_flow
                                
                target_view[('sample', frame_id, scale)] = sample
                target_view[('color', frame_id, scale)] = F.grid_sample(inputs[('color', frame_id, source_scale)][:, cam, ...], sample, padding_mode='border', align_corners=True)
                
                target_view[('warp_depth', frame_id, scale)] = self.forward_splat_depth_with_flow(
                            depth=ref_depth,  # (B,1,H,W)
                            flow=pred_flow,   # (B,2,H,W)
                            height=384,
                            width=640)
                            
                
                new_color = target_view[('color', frame_id, scale)][0].permute(1, 2, 0).detach().cpu().numpy()
                plt.figure(figsize=(20, 10))
                plt.imshow(new_color)
                plt.title(f"CAM{cam}, frame_id={frame_id}")
                plt.axis('off')
                plt.savefig('new_color.png')
            '''    
        outputs[('cam', cam)] = target_view

