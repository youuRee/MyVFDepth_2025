# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_util import Projection


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

    # t시점 예측한 depth을 가지고 t'시점 image에 
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
            ref_gt_depth = inputs[('gt_depth', 0)][:,cam, ...]
            ref_gt_depth_mask = (ref_gt_depth == 0) # 0인 부분만 True -> gt depth가 있는 픽셀은 RGB loss 반영 X
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
                
                # for temporal lidar loss
                # 같은 카메라라서 src, ref는 같은 내부 파라미터를 가짐
                gt_src_depth = inputs[('gt_depth', frame_id)][:,cam, ...]              
                src_invK = inputs[('inv_K', source_scale)][:,cam, ...]
                src_K = inputs[('K', source_scale)][:,cam, ...]
                
                # current view to the novel view
                warped_depth, warped_depth_mask = self.get_virtual_sparse_depth(
                        gt_src_depth, 
                        src_mask, 
                        ref_depth, 
                        ref_invK, 
                        ref_K, 
                        T,
                        self.min_depth,
                        self.max_depth
                    )

                target_view[('warp_depth', frame_id, scale)] = warped_depth
                target_view[('warp_depth_mask', frame_id, scale)] = warped_depth_mask
                
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
                    overlap_depth = torch.zeros_like(ref_gt_depth)
                    overlap_depth_mask = torch.zeros_like(ref_mask)

                    # cvcdepth
                    use_depth_consistency = hasattr(self, 'spatial_depth_consistency_loss_weight')
                    if use_depth_consistency:
                        overlap_depth2 = torch.zeros_like(ref_depth)
                    
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
                        
                        # for spatio lidar loss
                        gt_src_depth = inputs[('gt_depth', frame_id)][:, cur_index, ...]
                        warped_depth, warped_depth_mask = self.get_virtual_sparse_depth(
                                gt_src_depth, 
                                src_mask, 
                                ref_depth, 
                                ref_invK, 
                                src_K,
                                rel_pose, 
                                self.min_depth,
                                self.max_depth
                            )
                        
                        overlap_depth = overlap_depth + warped_depth
                        overlap_depth_mask = overlap_depth_mask + warped_depth_mask
                    
                        # cvcdepth
                        if use_depth_consistency:
                            if frame_id==0:
                                if self.spatial_depth_consistency_type=='pre':
                                    src_depth = outputs[('cam', cur_index)][('depth', scale)]
                                    src_invK = inputs[('inv_K', source_scale)][:, cur_index, ...]
                                    src_depth_tar_view = self.project.transform_depth(src_depth,torch.linalg.inv(rel_pose),src_invK,ref_K)[:,2:,]
                                    warped_depth2, warped_mask2 = self.get_virtual_image(
                                        src_depth_tar_view,
                                        src_mask,
                                        ref_depth,
                                        ref_invK,
                                        src_K,
                                        rel_pose,
                                        source_scale
                                    )

                                else:
                                    raise NotImplementedError


                                overlap_depth2 = overlap_depth2 + warped_depth2

                    target_view[('overlap', frame_id, scale)] = overlap_img
                    target_view[('overlap_mask', frame_id, scale)] = overlap_mask
                    
                    target_view[('overlap_depth', frame_id, scale)] = overlap_depth
                    target_view[('overlap_depth_mask', frame_id, scale)] = overlap_depth_mask

                    # cvcdepth
                    if use_depth_consistency:
                        target_view[('overlap_depth2', frame_id, scale)] = overlap_depth2
                    
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
                        
                rel_cam_list =  defaultdict(<class 'list'>, {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]})
                curr_index =  1
                curr_index =  2
                curr_index =  0
                rel_cam_list =  defaultdict(<class 'list'>, {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]})
                curr_index =  0
                curr_index =  3
                curr_index =  1
                rel_cam_list =  defaultdict(<class 'list'>, {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]})
                curr_index =  0
                curr_index =  4
                curr_index =  2
                rel_cam_list =  defaultdict(<class 'list'>, {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]})
                curr_index =  1
                curr_index =  5
                curr_index =  3
                rel_cam_list =  defaultdict(<class 'list'>, {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]})
                curr_index =  2
                curr_index =  5
                curr_index =  4
                rel_cam_list =  defaultdict(<class 'list'>, {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]})
                curr_index =  3
                curr_index =  4
                curr_index =  5
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

                    tform_depth.append(warp_depth)
                    tform_mask.append(warp_mask)

                target_view[('tform_depth', scale)] = tform_depth
                target_view[('tform_depth_mask', scale)] = tform_mask

        outputs[('cam', cam)] = target_view