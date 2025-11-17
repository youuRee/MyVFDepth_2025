# SECOND VFE
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix

from .blocks import conv2d, conv1d, pack_cam_feat
from utils import aug_depth_params

from cumm import tensorview as tv
from spconv.utils import Point2VoxelCPU3d
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch import functional as Fsp
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.hash import HashTable
import spconv.pytorch as spconv

from .spcnn import SpconvNet
from .voxel_encoder import VoxelFeatureExtractor
#from .vfe import VoxelFeatureExtractionKPconv, VoxelFeatureExtraction
import numpy as np

from typing import Tuple
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os 

def voxelize(
        points: torch.Tensor,
        voxel_size: torch.Tensor,
        grid_range: torch.Tensor,
        max_points_in_voxel: int,
        max_num_voxels: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts 3D point cloud to a sparse voxel grid using PyTorch.
    :param points: (num_points, num_features), first 3 elements must be <x>, <y>, <z>
    :param voxel_size: (3,) - <width>, <length>, <height>
    :param grid_range: (6,) - <min_x>, <min_y>, <min_z>, <max_x>, <max_y>, <max_z>
    :param max_points_in_voxel:
    :param max_num_voxels:
    :return: tuple (
        voxels (num_voxels, max_points_in_voxels, num_features),
        coordinates (num_voxels, 3),
        num_points_per_voxel (num_voxels,)
    )
    """
    points_copy = points.clone()
    grid_size = torch.floor((grid_range[3:] - grid_range[:3]) / voxel_size).to(torch.int32)

    coor_to_voxelidx = torch.full((grid_size[2], grid_size[1], grid_size[0]), -1, dtype=torch.int32, device=points.device)
    voxels = torch.zeros((max_num_voxels, max_points_in_voxel, points.shape[-1]), dtype=points_copy.dtype, device=points.device)
    coordinates = torch.zeros((max_num_voxels, 3), dtype=torch.int32, device=points.device)
    num_points_per_voxel = torch.zeros(max_num_voxels, dtype=torch.int32, device=points.device)

    points_coords = torch.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).to(torch.int32)
    mask = ((points_coords >= 0) & (points_coords < grid_size)).all(dim=1)
    points_coords = points_coords[mask][:, [2, 1, 0]]  # Swap dimensions to match the original order
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == points_coords.shape[0]

    voxel_num = 0
    for i, coord in enumerate(points_coords):
        voxel_idx = coor_to_voxelidx[tuple(coord)]
        if voxel_idx == -1:
            voxel_idx = voxel_num
            if voxel_num >= max_num_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[tuple(coord)] = voxel_idx
            coordinates[voxel_idx] = coord
        point_idx = num_points_per_voxel[voxel_idx]
        if point_idx < max_points_in_voxel:
            voxels[voxel_idx, point_idx] = points_copy[i]
            num_points_per_voxel[voxel_idx] += 1

    return voxels[:voxel_num], coordinates[:voxel_num], num_points_per_voxel[:voxel_num]


class MLP_empty(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_empty, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        return x

class MLP_unknown(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_unknown, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc3(x)
        return x

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels[0], kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(out_channels[0], out_channels[1], kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels[0])
        self.bn2 = nn.BatchNorm3d(out_channels[1])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class VFNet(nn.Module):
    """
    Surround-view fusion module that estimates a single 3D feature using surround-view images
    """
    def __init__(self, cfg, feat_in_dim, feat_out_dim, model='depth'):
        super(VFNet, self).__init__()
        self.read_config(cfg) 
        self.eps = 1e-8
        self.model = model

        self.sp_dim = 64
        self.sp_voxel = None

        self.spconv_net = SpconvNet(self.sp_dim)
        self.vfe = VoxelFeatureExtractor(num_input_features=3, num_filters=[32, 64], voxel_size=(1, 1, 1.5), pc_range=(-50, -50, -15, 49, 49, 13.5))
        
        # define the 3D voxel space(follows the DDAD extrinsic coordinate -- x: forward, y: left, z: up)
        # define voxel end range in accordance with voxel_str_p, voxel_size, voxel_unit_size
        self.voxel_end_p = [self.voxel_str_p[i] + self.voxel_unit_size[i] * (self.voxel_size[i] - 1) for i in range(3)]
            
        # define a voxel space, [1, 3, z, y, x], each voxel contains its 3D position
        voxel_grid = self.create_voxel_grid(self.voxel_str_p, self.voxel_end_p, self.voxel_size)        
        b, _, self.z_dim, self.y_dim, self.x_dim = voxel_grid.size()
        self.n_voxels = self.z_dim * self.y_dim * self.x_dim
        ones = torch.ones(self.batch_size, 1, self.n_voxels)
        self.voxel_pts = torch.cat([voxel_grid.view(b, 3, self.n_voxels), ones], dim=1)

        # define grids in pixel space
        self.img_h = self.height // (2 ** (self.fusion_level+1))
        self.img_w = self.width // (2 ** (self.fusion_level+1))
        self.num_pix = self.img_h * self.img_w
        self.pixel_grid = self.create_pixel_grid(self.batch_size, self.img_h, self.img_w)
        self.pixel_ones = torch.ones(self.batch_size, 1, self.proj_d_bins, self.num_pix)
        
        # define a depth grid for projection
        depth_bins = torch.linspace(self.proj_d_str, self.proj_d_end, self.proj_d_bins)
        self.depth_grid = self.create_depth_grid(self.batch_size, self.num_pix, self.proj_d_bins, depth_bins)
        
        # depth fusion(process overlap and non-overlap regions)
        if model == 'depth':
            # voxel - preprocessing layer
            self.v_dim_o = [(feat_in_dim + 1) * 2] + self.voxel_pre_dim
            self.v_dim_no = [feat_in_dim + 1] + self.voxel_pre_dim
            
            self.conv_overlap = conv1d(self.v_dim_o[0], self.v_dim_o[1], kernel_size=1) 
            self.conv_non_overlap = conv1d(self.v_dim_no[0], self.v_dim_no[1], kernel_size=1) 

            encoder_dims = self.proj_d_bins * (self.v_dim_o[-1] + self.sp_dim)
            stride = 1
            
        else:
            encoder_dims = (self.sp_dim + feat_in_dim + 1)*self.z_dim
            stride = 2
            
        # channel dimension reduction
        self.reduce_dim = nn.Sequential(*conv2d(encoder_dims, 256, kernel_size=3, stride = stride).children(),
                                        *conv2d(256, feat_out_dim, kernel_size=3, stride = stride).children())          

        #self.conv3d = Conv3DBlock(in_channels=128, out_channels=[128, 64], kernel_size=3, stride=1, padding=1)
        self.conv3d = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv3d_block = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),  # 3x3x3 3D Convolution
            nn.BatchNorm3d(128),  # 3D Batch Normalization
            nn.ReLU(),  # ReLU activation
            #nn.Conv3d(128, 256, kernel_size=3, padding=1),  # 3x3x3 3D Convolution
            #nn.BatchNorm3d(256),  # 3D Batch Normalization
            #nn.ReLU(),  # ReLU activation
            #nn.Conv3d(256, 128, kernel_size=3, padding=1),  # 3x3x3 3D Convolution
            #nn.BatchNorm3d(128),  # 3D Batch Normalization
            #nn.ReLU(),  # ReLU activation
            nn.Conv3d(128, 64, kernel_size=3, padding=1),  # 3x3x3 3D Convolution
            nn.BatchNorm3d(64),  # 3D Batch Normalization
            nn.ReLU()
        )

        self.save_cnt = 0

        # MLP1: 3차원 좌표 -> 은닉층 1 -> 출력
        self.emptynet = MLP_empty(input_dim=1,  output_dim=64)

        # MLP2: 3차원 좌표 -> 은닉층 1 -> 은닉층 2 -> 출력
        self.unknownnet = MLP_unknown(input_dim=1, hidden_dim=32, output_dim=64)

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
    
    def create_voxel_grid(self, str_p, end_p, v_size):
        """
        output: [batch, 3, z_dim, y_dim, x_dim]
        [b, :, z, y, x] contains (x,y,z) 3D point
        """
        grids = [torch.linspace(str_p[i], end_p[i], v_size[i]) for i in range(3)]

        x_dim, y_dim, z_dim = v_size
        grids[0] = grids[0].view(1, 1, 1, 1, x_dim)
        grids[1] = grids[1].view(1, 1, 1, y_dim, 1)
        grids[2] = grids[2].view(1, 1, z_dim, 1, 1)
        
        grids = [grid.expand(self.batch_size, 1, z_dim, y_dim, x_dim) for grid in grids]
        return torch.cat(grids, 1)
    
    def create_pixel_grid(self, batch_size, height, width):
        """
        output: [batch, 3, height * width]
        """
        grid_xy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        pix_coords = torch.stack(grid_xy, axis=0).unsqueeze(0).view(1, 2, height * width)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        ones = torch.ones(batch_size, 1, height * width)
        pix_coords = torch.cat([pix_coords, ones], 1)
        return pix_coords

    def create_depth_grid(self, batch_size, n_pixels, n_depth_bins, depth_bins):
        """
        output: [batch, 3, num_depths, height * width]
        """
        depth_layers = []
        for d in depth_bins:
            depth_layer = torch.ones((1, n_pixels)) * d
            depth_layers.append(depth_layer)
        depth_layers = torch.cat(depth_layers, dim=0).view(1, 1, n_depth_bins, n_pixels)
        depth_layers = depth_layers.expand(batch_size, 3, n_depth_bins, n_pixels)
        return depth_layers

    def type_check(self, sample_tensor):
        """
        This function checks the type of the tensor, so that all the parameters share same device and dtype.
        """
        d_dtype, d_device = sample_tensor.dtype, sample_tensor.device
        if (self.voxel_pts.dtype != d_dtype) or (self.voxel_pts.device != d_device):
            self.voxel_pts = self.voxel_pts.to(device=d_device, dtype=d_dtype)
            self.pixel_grid = self.pixel_grid.to(device=d_device, dtype=d_dtype)
            self.depth_grid = self.depth_grid.to(device=d_device, dtype=d_dtype)
            self.pixel_ones = self.pixel_ones.to(device=d_device, dtype=d_dtype)    

    def backproject_into_voxel(self, feats_agg, input_mask, intrinsics, extrinsics_inv):
        """
        This function backprojects 2D features into 3D voxel coordinate using intrinsic and extrinsic of each camera.
        Self-occluded regions are removed by using the projected mask in 3D voxel coordinate.
        """
        voxel_feat_list = []
        voxel_mask_list = []
        
        for cam in range(self.num_cams):
            feats_img = feats_agg[:, cam, ...]
            _, _, h_dim, w_dim = feats_img.size()
        
            mask_img = input_mask[:, cam, ...]            
            mask_img = F.interpolate(mask_img, [h_dim, w_dim], mode='bilinear', align_corners=True)            
            
            # 3D points in the voxel grid -> 3D points referenced at each view. [b, 3, n_voxels]
            ext_inv_mat = extrinsics_inv[:, cam, :3, :]
            v_pts_local = torch.matmul(ext_inv_mat, self.voxel_pts)

            # calculate pixel coordinate that each point are projected in the image. [b, n_voxels, 1, 2]
            K_mat = intrinsics[:, cam, :, :]
            pix_coords = self.calculate_sample_pixel_coords(K_mat, v_pts_local, w_dim, h_dim)

            # compute validity mask. [b, 1, n_voxels]
            valid_mask = self.calculate_valid_mask(mask_img, pix_coords, v_pts_local)
            
            # retrieve each per-pixel feature. [b, feat_dim, n_voxels, 1]
            feat_warped = F.grid_sample(feats_img, pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)            
            # concatenate relative depth as the feature. [b, feat_dim + 1, n_voxels]
            feat_warped = torch.cat([feat_warped.squeeze(-1), v_pts_local[:, 2:3, :]/(self.voxel_size[0])], dim=1)
            feat_warped = feat_warped * valid_mask.float()
            
            voxel_feat_list.append(feat_warped)
            voxel_mask_list.append(valid_mask)
        
        # compute overlap region
        voxel_mask_count = torch.sum(torch.cat(voxel_mask_list, dim=1), dim=1, keepdim=True)
        
        if self.model == 'depth':
            # discriminatively process overlap and non_overlap regions using different MLPs
            voxel_non_overlap = self.preprocess_non_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
            voxel_overlap = self.preprocess_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
            voxel_feat = voxel_non_overlap + voxel_overlap
            
        elif self.model == 'pose':
            voxel_feat = torch.sum(torch.stack(voxel_feat_list, dim=1), dim=1, keepdim=False)
            voxel_feat = voxel_feat/(voxel_mask_count+1e-7)
            
        return voxel_feat

    def calculate_sample_pixel_coords(self, K, v_pts, w_dim, h_dim):
        """
        This function calculates pixel coords for each point([batch, n_voxels, 1, 2]) to sample the per-pixel feature.
        """        
        cam_points = torch.matmul(K[:, :3, :3], v_pts)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)

        if not torch.all(torch.isfinite(pix_coords)):
            pix_coords = torch.clamp(pix_coords, min=-w_dim*2, max=w_dim*2)

        pix_coords = pix_coords.view(self.batch_size, 2, self.n_voxels, 1)
        pix_coords = pix_coords.permute(0, 2, 3, 1) 
        pix_coords[:, :, :, 0] = pix_coords[:, :, :, 0] / (w_dim - 1)
        pix_coords[:, :, :, 1] = pix_coords[:, :, :, 1] / (h_dim - 1)
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

    def calculate_valid_mask(self, mask_img, pix_coords, v_pts_local):
        """
        This function creates valid mask in voxel coordinate by projecting self-occlusion mask to 3D voxel coords. 
        """
        # compute validity mask, [b, 1, n_voxels, 1]
        mask_selfocc = (F.grid_sample(mask_img, pix_coords, mode='nearest', padding_mode='zeros', align_corners=True) > 0.5)
        # discard points behind the camera, [b, 1, n_voxels]
        mask_depth = (v_pts_local[:, 2:3, :] > 0) 
        # compute validity mask, [b, 1, n_voxels, 1]
        pix_coords_mask = pix_coords.permute(0, 3, 1, 2)
        mask_oob = ~(torch.logical_or(pix_coords_mask > 1, pix_coords_mask < -1).sum(dim=1, keepdim=True) > 0)
        valid_mask = mask_selfocc.squeeze(-1) * mask_depth * mask_oob.squeeze(-1)
        return valid_mask
    
    def preprocess_non_overlap(self, voxel_feat_list, voxel_mask_list, voxel_mask_count):
        """
        This function applies 1x1 convolutions to features from non-overlapping features.
        """
        non_overlap_mask = (voxel_mask_count == 1)
        voxel = sum(voxel_feat_list)
        voxel = voxel * non_overlap_mask.float()

        for conv_no in self.conv_non_overlap:
            voxel = conv_no(voxel)
        return voxel * non_overlap_mask.float()

    def preprocess_overlap(self, voxel_feat_list, voxel_mask_list, voxel_mask_count):
        """
        This function applies 1x1 convolutions on overlapping features.
        Camera configuration [0,1,2] or [0,1,2,3,4,5]:
                        3 1
            rear cam <- 5   0 -> front cam
                        4 2
        """
        overlap_mask = (voxel_mask_count == 2)
        if self.num_cams == 3:
            feat1 = voxel_feat_list[0]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2]
        elif self.num_cams == 6:
            feat1 = voxel_feat_list[0] + voxel_feat_list[3] + voxel_feat_list[4]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2] + voxel_feat_list[5]
        else:
            raise NotImplementedError
            
        voxel = torch.cat([feat1, feat2], dim=1)
        for conv_o in self.conv_overlap:
            voxel = conv_o(voxel)
        return voxel * overlap_mask.float()

    def project_voxel_into_image(self, cam_voxel_feat, lidar_voxel_feat, inv_K, extrinsics):
        """
        This function projects voxels into 2D image coordinate. 
        [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        """        
        # define depth bin
        # [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        b, feat_dim, _ = cam_voxel_feat.size()
        cam_voxel_feat = cam_voxel_feat.view(b, feat_dim, self.z_dim, self.y_dim, self.x_dim)

        #lidar_voxel_feat = lidar_voxel_feat * prob_voxel
        voxel_feat = torch.cat((cam_voxel_feat, lidar_voxel_feat), dim=1) # [1, 128, 20, 100, 100]
        
        
        #bev_feat = voxel_feat.mean(dim=2)
        #bev_mean = bev_feat.mean(dim=1)[0].detach().cpu().numpy()
        #plt.imshow(bev_mean, cmap='viridis')
        #plt.savefig('/workspace/VFDepth/voxel_bev.png')
        #plt.savefig(os.path.join('/workspace/VFDepth/DL6_weights_17/bev/', '{}'.format(self.save_cnt) + ".png"))
        #self.save_cnt += 1
        

        proj_feats = []
        for cam in range(self.num_cams):
            # construct 3D point grid for each view
            cam_points = torch.matmul(inv_K[:, cam, :3, :3], self.pixel_grid)
            cam_points = self.depth_grid * cam_points.view(self.batch_size, 3, 1, self.num_pix) # [b, 3, n_depthbins, n_pixels]
            cam_points = torch.cat([cam_points, self.pixel_ones], dim=1) # [b, 3, n_depthbins, n_pixels] [b, 1, n_depthbins, n_pixels] -> [b, 4, n_depthbins, n_pixels]
            cam_points = cam_points.view(self.batch_size, 4, -1) # [b, 4, n_depthbins * n_pixels]
            
            # apply extrinsic: local 3D point -> global coordinate, [b, 3, n_depthbins * n_pixels]
            points = torch.matmul(extrinsics[:, cam, :3, :], cam_points)

            # 3D grid_sample [b, n_voxels, 3], value: (x, y, z) point
            grid = points.permute(0, 2, 1) 
            
            for i in range(3):
                v_length = self.voxel_end_p[i] - self.voxel_str_p[i]
                grid[:, :, i] = (grid[:, :, i] - self.voxel_str_p[i]) / v_length * 2. - 1.
                
            grid = grid.view(self.batch_size, self.proj_d_bins, self.img_h, self.img_w, 3)
            # 이쪽 부분에서 LiDAR point 없는 빈 voxel 처리 or LiDAR point 있는 voxel을 잘 활용
            # ex) sp_dim에 mask 적용 (point 존재 여부, 카메라 범위 밖)
            # ex) LiDAR point 있는 voxel에 해당하는 depth bin에 높은 확률 부여 (softmax -> LLS 참고)
            
            proj_feat = F.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True) # (b,c,bin,h,w), [1, 128, 50, 48, 80])
            proj_feat = proj_feat.view(b, self.proj_d_bins * (self.v_dim_o[-1] + self.sp_dim), self.img_h, self.img_w) # [1, 6400, 48, 80]
            
            # conv, reduce dimension
            proj_feat = self.reduce_dim(proj_feat)
            proj_feats.append(proj_feat)
        '''
        proj_mean = proj_feats[0].mean(dim=1) 

        # 이미지를 저장 (1채널 이미지로 저장)
        vutils.save_image(proj_mean, '/workspace/VFDepth/proj_prob_voxel.png')

        # 채널 방향으로 평균을 계산하여 (25, 25) 크기의 2D 배열로 변환
        proj_mean = proj_feats[0].mean(dim=1)[0].detach().cpu().numpy()

        # 시각화 후 이미지 파일로 저장
        plt.imshow(proj_mean, cmap='viridis')

        # 이미지 파일로 저장 (PNG 포맷)
        plt.savefig('/workspace/VFDepth/proj_prob_voxel_plot.png')
        '''
        return proj_feats

    def augment_extrinsics(self, ext):
        """
        This function augments depth estimation results using augmented extrinsics [batch, cam, 4, 4]  
        """
        with torch.no_grad():
            b, cam, _, _ = ext.size()
            ext_aug = ext.clone()

            # rotation augmentation
            angle = torch.rand(b, cam, 3)
            for i in range(3):
                angle[:, :, i] = (angle[:, :, i] - 0.5) * self.aug_angle[i]
            angle_mat = axis_angle_to_matrix(angle) # 3x3
            tform_mat = torch.eye(4).repeat(b, cam, 1, 1)
            tform_mat[:, :, :3, :3] = angle_mat
            tform_mat = tform_mat.to(device=ext.device, dtype=ext.dtype)

            ext_aug = tform_mat @ ext_aug
        return ext_aug
    

    def VoxelRayCasting(self, coords, grid_range, voxel_size, voxelwise):
        voxel_grid = self.create_voxel_grid(self.voxel_str_p, self.voxel_end_p, self.voxel_size).to('cuda')
        free_voxel = torch.ones([1, 64, 20, 100, 100]).to('cuda')
        free_voxel2 = torch.zeros([1, 1, 20, 100, 100]).to('cuda')

        # 벡터화된 방식으로 차있는 복셀을 한 번에 할당
        v_z, v_y, v_x = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
        free_voxel[:, :, v_z, v_y, v_x] = voxelwise.T.unsqueeze(0)  # 차있는 복셀 -> voxelwise 값
        free_voxel2[:, :, v_z, v_y, v_x] = 2
        empty_feat = None
        # 샘플링된 점 계산 및 복셀 값 할당
        ray_origin = torch.tensor([0, 0, 0], dtype=torch.float32).to('cuda')

        # 방향 벡터 및 샘플링 점 벡터화
        for i in range(coords.shape[0]):
            v_z, v_y, v_x = coords[i].long()
            target = voxel_grid[:, :, v_z, v_y, v_x].squeeze(0) + torch.tensor([0.5, 0.5, 0.75], dtype=torch.float32).to('cuda')
            direction = (target - ray_origin) / torch.norm(target - ray_origin)

            # 샘플링된 점 계산
            num_points = 50
            total_distance = torch.norm(target - ray_origin)
            distances = torch.linspace(0, total_distance, steps=num_points + 1)[1:].to('cuda')
            sampled_points = ray_origin.unsqueeze(0) + distances.unsqueeze(1) * direction.unsqueeze(0)

            # 샘플링 점의 복셀 coords (벡터화)
            sample_points_coords = torch.floor((sampled_points[:, :3] - grid_range[:3]) / voxel_size)
            sample_points_coords = sample_points_coords[:, [2, 1, 0]].int()

            # 벡터화된 방식으로 포인트가 없는 복셀에 대한 처리
            matches = torch.all(coords.unsqueeze(1) == sample_points_coords.unsqueeze(0), dim=2)  # 벡터화된 일치 확인
            empty_indices = torch.nonzero(matches.sum(0) == 0)  # 일치하지 않는 복셀만 추출
            #print(empty_indices.shape) # torch.Size([2, 1])
            #print(empty_indices) # tensor([[0],
                                 #        [3]], device='cuda:0')
            if empty_indices.shape[0] > 0:  # 비어있는 복셀에 대한 처리
                empty_coords = sample_points_coords[empty_indices[:, 0]]  # empty_indices에서 실제 좌표를 추출
                z, y, x = empty_coords[:, 0].long(), empty_coords[:, 1].long(), empty_coords[:, 2].long()
                if empty_feat is None:
                    empty_feat = self.emptynet(torch.tensor([1.0], device='cuda'))
                #print(free_voxel[:, :, z, y, x].shape) # (1, 64, 2(empty 복셀 개수))
                free_voxel[:, :, z, y, x] = empty_feat.unsqueeze(0).unsqueeze(2).expand(1, 64, empty_indices.shape[0])
                free_voxel2[:, :, z, y, x] = 1

        # unknown_idx를 벡터화된 방식으로 처리
        unknown_idx = torch.nonzero(free_voxel[0,0,:,:,:] == 1)
        unknown_feat = self.unknownnet(torch.tensor([1.0], device='cuda'))

        # 벡터화된 방식으로 unknown_feat 할당
        if unknown_idx.shape[0] > 0:
            z, y, x = unknown_idx[:, 0].long(), unknown_idx[:, 1].long(), unknown_idx[:, 2].long()
            free_voxel[:, :, z, y, x] = unknown_feat.unsqueeze(0).unsqueeze(2).expand(1, 64, unknown_idx.shape[0])

        return free_voxel


    def LiDARFeatureExtractor(self, lidar):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        voxel_size = torch.tensor([1, 1, 1.5], dtype=torch.float32).to('cuda')
        grid_range = torch.tensor([-50, -50, -15, 49, 49, 13.5], dtype=torch.float32).to('cuda')
        #voxels, coords, num_points_in_voxel = voxelize(lidar, voxel_size, grid_range, max_points_in_voxel=5, max_num_voxels=200000)
        
        voxel_gen = PointToVoxel(vsize_xyz=[1, 1, 1.5],
                                coors_range_xyz=[-50, -50, -15, 49, 49, 13.5],
                                num_point_features=3,
                                max_num_voxels=200000,
                                max_num_points_per_voxel=5,
                                device=device
                                )
        
        voxels, coords, num_points_in_voxel, pc_voxel_id = voxel_gen.generate_voxel_with_id(lidar)
        voxelwise = self.vfe(voxels, num_points_in_voxel, coords)#.clone().detach()
        #features = torch.mean(voxels, dim=1).to(device)
        voxel_coordinates = coords.to(torch.int32).to(device)
        
        if self.model == 'depth':
            free_voxel = self.VoxelRayCasting(voxel_coordinates, grid_range, voxel_size, voxelwise)
            voxel_feat = self.conv3d_block(free_voxel)
        else:

            batch_indices = torch.zeros((voxelwise.shape[0], self.batch_size), dtype=torch.int32).to(device)
            input_indices = torch.cat((batch_indices, voxel_coordinates), dim=1)

            input_sp_tensor = spconv.SparseConvTensor(
                features=voxelwise,
                indices=input_indices,
                spatial_shape=torch.tensor([20, 100, 100]).to(device),
                batch_size=self.batch_size
            )
            
            
            # 포워드 패스를 수행하여 복셀 특징을 얻기
            output_sp_tensor = self.spconv_net(input_sp_tensor)
            voxel_feat = output_sp_tensor.dense()

        return voxel_feat

    
    def forward(self, inputs, feats_agg, frame_ids):
        
        mask = inputs['mask']
        K = inputs['K', self.fusion_level+1]
        inv_K = inputs['inv_K', self.fusion_level+1]
        extrinsics = inputs['extrinsics']
        extrinsics_inv = inputs['extrinsics_inv']
        free_voxel = inputs['voxel_label'][:,0,:,:,:,:]
        
        #lidar = inputs['point_cloud']
        #print(inputs['voxels'].shape) # [1, 6, 5052, 5, 3]
        #print(inputs['coords'].shape) # [1, 6, 5052, 3]
        #print(inputs['voxel_occ'].shape) # [1, 6, 20, 100, 100]
        fusion_dict = {}
        for cam in range(self.num_cams):
            fusion_dict[('cam', cam)] = {}
        
        # device, dtype check, match dtype and device
        sample_tensor = feats_agg[0, 0, ...] # B, n_cam, c, h, w
        self.type_check(sample_tensor)
            
        # backproject each per-pixel feature into 3D space (or sample per-pixel features for each voxel)
        cam_voxel_feat = self.backproject_into_voxel(feats_agg, mask, K, extrinsics_inv)
        #print("voxel_feat: ", voxel_feat.shape)
        #self.spconvTensor(lidar[0][0])
        
        if self.model == 'depth':
            pc = inputs[('lidar', frame_ids)]
            lidar_voxel_feat = self.LiDARFeatureExtractor(pc[0][0])
            # for each pixel, collect voxel features -> output image feature     
            proj_feats = self.project_voxel_into_image(cam_voxel_feat, lidar_voxel_feat, inv_K, extrinsics)
            fusion_dict['proj_feat'] = pack_cam_feat(torch.stack(proj_feats, 1))
 
            # with view augmentation
            if self.aug_depth:
                # extrinsics
                inputs['extrinsics_aug'] = self.augment_extrinsics(extrinsics)
                proj_feats = self.project_voxel_into_image(cam_voxel_feat, lidar_voxel_feat, inv_K, inputs['extrinsics_aug'])
                fusion_dict['proj_feat_aug'] = pack_cam_feat(torch.stack(proj_feats, 1))

            # synthesis visualization
            if self.syn_visualize:
                def _get_proj_feat(inv_K, ang_x, ang_y, ang_z):
                    angle_mat = axis_angle_to_matrix(torch.tensor([ang_x, ang_y, ang_z])[None, :]) # 3x3
                    b, c, _, _ = extrinsics.size()
                    tform_mat = torch.eye(4)[None, None]
                    tform_mat[:, :, :3, :3] = angle_mat
                    tform_mat = tform_mat.repeat(b, c, 1, 1).to(device=extrinsics.device, dtype=extrinsics.dtype)
                    proj_feats = self.project_voxel_into_image(cam_voxel_feat, lidar_voxel_feat,  inv_K, tform_mat @ extrinsics)
                    return proj_feats[0]

                fusion_dict['syn_feat'] = []
                
                # augmented intrinsics and extrinsics
                aug_params = aug_depth_params(K)   
                for param in aug_params:
                    fusion_dict['syn_feat'] += [_get_proj_feat(*param)]  
            return fusion_dict      
        
        elif self.model == 'pose':
            cur_pc = inputs[('lidar', frame_ids[0])] 
            next_pc = inputs[('lidar', frame_ids[1])]
            cur_voxel = self.LiDARFeatureExtractor(cur_pc[0][0]) # [1, 64, 20, 100, 100]
            next_voxel = self.LiDARFeatureExtractor(next_pc[0][0])
            
            b, c, _, _, _ = cur_voxel.shape
            cur_voxel = cur_voxel.view(b, c, self.z_dim * self.y_dim * self.x_dim) # [1, 64, 200000]
            next_voxel = next_voxel.view(b, c, self.z_dim * self.y_dim * self.x_dim)  
            lidar_voxel = torch.cat([cur_voxel, next_voxel], 1).view(b, c*2, self.z_dim, self.y_dim, self.x_dim) # [1, 128, 200000] -> [1, 128, 20, 100, 100]    
            lidar_voxel = self.conv3d(lidar_voxel) # [1, 64, 20, 100, 100]
            lidar_voxel = lidar_voxel.view(b, 64, self.z_dim * self.y_dim * self.x_dim)
            voxel_feat = torch.cat([cam_voxel_feat, lidar_voxel], 1)

            b, c, _ = voxel_feat.shape # torch.Size([1, 257, 200000])     
            voxel_feat = voxel_feat.view(b, c*self.z_dim, 
                                         self.y_dim, self.x_dim)            
            bev_feat = self.reduce_dim(voxel_feat) # torch.Size([1, 128, 25, 25])

            '''
            # 채널 방향으로 평균을 계산하여 (25, 25) 크기의 2D 배열로 변환
            # BEV feature 텐서의 채널 평균을 계산
            bev_mean = bev_feat.mean(dim=1)  # (1, 25, 25) 크기의 텐서

            # 이미지를 저장 (1채널 이미지로 저장)
            vutils.save_image(bev_mean, '/workspace/VFDepth/bev_channel_mean.png')

            # 채널 방향으로 평균을 계산하여 (25, 25) 크기의 2D 배열로 변환
            bev_mean = bev_feat.mean(dim=1)[0].detach().cpu().numpy()

            # 시각화 후 이미지 파일로 저장
            plt.imshow(bev_mean, cmap='viridis')

            # 이미지 파일로 저장 (PNG 포맷)
            plt.savefig('/workspace/VFDepth/bev_channel_mean_plot.png')
            '''
            return bev_feat