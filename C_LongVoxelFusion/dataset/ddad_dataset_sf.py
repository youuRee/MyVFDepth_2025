# Copyright (c) 2023 42dot. All rights reserved.
import os
import numpy as np
import pandas as pd

from .data_util import transform_mask_sample, mask_loader_scene, align_dataset

from external.utils import Camera, generate_depth_map, make_list
from external.dataset import DGPDataset, SynchronizedSceneDataset, stack_sample
from itertools import chain

from pyquaternion import Quaternion
import torch

import matplotlib.pyplot as plt
import imageio

class DDADdatasetSF(DGPDataset):
    """
    Superclass for DGP dataset loaders of the packnet-sfm repository.
    """
    def __init__(self, *args, with_mask, scale_range, **kwargs):
        super().__init__(*args, **kwargs)
        self.cameras = kwargs['cameras']
        self.scales = np.arange(scale_range+2) 

        ## self-occ masks 
        self.with_mask = with_mask
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.mask_path = os.path.join(cur_path, 'ddad_mask')
        file_name = os.path.join(self.mask_path, 'mask_idx_dict.pkl')
        
        self.mask_idx_dict = pd.read_pickle(file_name)
        self.mask_loader = mask_loader_scene
        
        datum_names = self.cameras + ['lidar']
        self.dataset = SynchronizedSceneDataset(self.path,
                        split=self.split,
                        datum_names=datum_names,
                        backward_context=self.bwd,
                        forward_context=self.fwd,
                        requested_annotations=None,
                        only_annotated_datums=False,
                        )        

    def generate_depth_map_sf(self, sample_idx, datum_idx, filename):
        """
        This function follows structure of dgp_dataset/generate_depth_map in packnet-sfm. 
        Due to the version issue with dgp, minor revision was made to get the correct value.
        """      
        # generate depth filename
        filename = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('depth/{}'.format(self.depth_type)))

        # image 파일 이름 생성
        #depth_filename_image = '{}/{}.png'.format('/workspace/VFDepth/', filename.format('depth/{}'.format(self.depth_type)))
        
        # load and return if exists
        if os.path.exists(filename):
            return np.load(filename, allow_pickle=True)['depth']
        # otherwise, create, save and return
        else:
            # get pointcloud
            scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
            pc_datum_data, _ = self.dataset.get_point_cloud_from_datum(
                                scene_idx, sample_idx_in_scene, self.depth_type)
            # test: 48 Channel Depth Map train: 16 Channel Depth Map
            #pc_datum_data['point_cloud'] = self.point_downsampling(pc_datum_data['point_cloud'], 48)
            #print(scene_idx, 'Depth Map', pc_datum_data['point_cloud'].shape)
            # create camera
            camera_rgb = self.get_current('rgb', datum_idx)
            camera_pose = self.get_current('pose', datum_idx)
            camera_intrinsics = self.get_current('intrinsics', datum_idx)
            camera = Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())
            
            # generate depth map
            world_points = pc_datum_data['pose'] * pc_datum_data['point_cloud']
            depth = generate_depth_map(camera, world_points, camera_rgb.size[::-1])
            
            # save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)

            # 깊이 맵을 이미지로 저장
            #os.makedirs(os.path.dirname(depth_filename_image), exist_ok=True)
            #imageio.imwrite(depth_filename_image, (depth * 255).astype(np.uint8))
            
            return depth
        
    # idx, cam, filename, points
    def generate_sparse_depth_map_sf(self, sample_idx, datum_idx, point):
        """
        This function follows structure of dgp_dataset/generate_depth_map in packnet-sfm. 
        Due to the version issue with dgp, minor revision was made to get the correct value.
        """
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
        pc_datum_data, _ = self.dataset.get_point_cloud_from_datum(
                            scene_idx, sample_idx_in_scene, 'lidar')
        
        #print(pc_datum_data['pose'])
        #print(pc_datum_data['point_cloud'])
        sparse_depth = []
        
        # 현재 시점(t) depth 저장
        camera_rgb = self.get_current('rgb', datum_idx)
        camera_pose = self.get_current('pose', datum_idx)
        camera_intrinsics = self.get_current('intrinsics', datum_idx)
        camera = Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())

        # generate depth map
        world_points = pc_datum_data['pose'] * pc_datum_data['point_cloud']
        depth = generate_depth_map(camera, world_points, camera_rgb.size[::-1])
        
        sparse_depth.append(depth)
        #imageio.imwrite('./curr_depth.png', (depth * 255).astype(np.uint8))
        #plt.imshow(depth)
        #plt.savefig("curr_depth.png")
        
        # 다른 시점(t-1, t+1) frame depth 저장
        lidar_pose = self.get_context('pose', 6)
        lidar_point = self.get_context('point_cloud', 6)
        
        camera_rgb = self.get_context('rgb', datum_idx)
        camera_pose = self.get_context('pose', datum_idx)
        camera_intrinsics = self.get_context('intrinsics', datum_idx)
        camera = Camera(K=camera_intrinsics[0], p_cw=camera_pose[0].inverse())
        
        world_points = np.dot(lidar_pose[0].matrix[:3,:3], lidar_point[0].T).T + lidar_pose[0].matrix[:3,3]
        depth = generate_depth_map(camera, world_points, camera_rgb[0].size[::-1])
        
        sparse_depth.append(depth)
        #imageio.imwrite('./prev_depth.png', (depth * 255).astype(np.uint8))
        #plt.imshow(depth)
        #plt.savefig("prev_depth.png")
        
        camera = Camera(K=camera_intrinsics[1], p_cw=camera_pose[1].inverse())
        world_points = np.dot(lidar_pose[1].matrix[:3,:3], lidar_point[1].T).T + lidar_pose[1].matrix[:3,3]
        depth = generate_depth_map(camera, world_points, camera_rgb[1].size[::-1])
        
        sparse_depth.append(depth)
        #imageio.imwrite('./next_depth.png', (depth * 255).astype(np.uint8))
        #plt.imshow(depth)
        #plt.savefig("next_depth.png")
        
        return sparse_depth
    
    def generate_range_img(self, point):
        
        # 각 점의 수직 각도 계산
        vertical_angles = np.arctan(point[:, 2] / np.sqrt(point[:, 0]**2 + point[:, 1]**2))

        # 라디안 값을 도로 변환
        pc_vertical_angles = np.degrees(vertical_angles)

        min_vertical_angle = 0
        max_vertical_angle = 30
        vertical_resolution = 0.03
        desired_rings = 64

        total_range = max_vertical_angle - min_vertical_angle
        new_vertical_resolution = total_range / (desired_rings - 1)

        # 0~30도에서 64개의 균일한 vertical angles
        ring_vertical_angles = np.linspace(min_vertical_angle, max_vertical_angle, desired_rings)

        
        vertical_idx = []
        for pc_angle in pc_vertical_angles:
            #closest_ring_angle = ring_vertical_angles[np.argmin(np.abs(ring_vertical_angles - pc_angle))]
            idx = np.argmin(np.abs(ring_vertical_angles - pc_angle))
            #rings.append(closest_ring_angle)
            vertical_idx.append(idx)

        min_horizontal_angle = 0
        max_horizontal_angle = 120
        horizontal_resolution = 0.07

        total_horizontal_range = (max_horizontal_angle / horizontal_resolution) * 4 # LiDAR 4개

        # 0~360도에서 total_horizontal_range개의 균일한 horizontal angles
        ring_horizontal_angles = np.linspace(-180, 180, np.round(total_horizontal_range).astype(int))
        horizontal_angles = np.arctan2(point[:, 1], point[:, 0])
        pc_horizontal_angles = np.degrees(horizontal_angles)

        horizontal_idx = []
        for pc_angle in pc_horizontal_angles:
            #closest_ring_angle = ring_vertical_angles[np.argmin(np.abs(ring_vertical_angles - pc_angle))]
            idx = np.argmin(np.abs(ring_horizontal_angles - pc_angle))
            #rings.append(closest_ring_angle)
            horizontal_idx.append(idx)
        
        np_vertical_idx = np.array([vertical_idx])
        np_horizontal_idx = np.array([horizontal_idx])
        coord = np.concatenate((np_vertical_idx, np_horizontal_idx)).T

        range_image = np.zeros((64, 6858)) # desired_rings, total_horizontal_range

        for i in range(point.shape[0]):
            d = np.sqrt(point[i][0]**2 + point[i][1]**2 + point[i][2]**2)
            if range_image[coord[i][0]][coord[i][1]] == 0:
                range_image[coord[i][0]][coord[i][1]] = d
            else:
                range_image[coord[i][0]][coord[i][1]] = min(d, range_image[coord[i][0]][coord[i][1]])
        
        return range_image, coord

            
    def get_filename_sf(self, sample_idx, datum_idx):
        """
        This function is defined to meet dgp version(v1.4)
        """
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
        scene_dir = self.dataset.scenes[scene_idx].directory
        filename = self.dataset.get_datum(
            scene_idx, sample_idx_in_scene, datum_indices[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def point_downsampling(self, point, ch):
        
        # 각 점의 수직 각도 계산
        vertical_angles = np.arctan(point[:, 2] / np.sqrt(point[:, 0]**2 + point[:, 1]**2))

        # 라디안 값을 도로 변환
        pc_vertical_angles = np.degrees(vertical_angles)

        min_vertical_angle = 0
        max_vertical_angle = 30
        vertical_resolution = 0.03
        desired_rings = 64

        total_range = max_vertical_angle - min_vertical_angle
        new_vertical_resolution = total_range / (desired_rings - 1)

        # 0~30도에서 64개의 균일한 vertical angles
        ring_vertical_angles = np.linspace(min_vertical_angle, max_vertical_angle, desired_rings)

        rings = []

        for pc_angle in pc_vertical_angles:
            closest_ring_angle = ring_vertical_angles[np.argmin(np.abs(ring_vertical_angles - pc_angle))]
            rings.append(closest_ring_angle)

        rings = np.array(rings)
        new_data = np.concatenate((point, rings[:, np.newaxis]), axis=1)

        # 기존 채널 64로 가정
        ring_64 = []
        for i in range(ring_vertical_angles.shape[0]):
            tmp = []
            for j in range(new_data.shape[0]):
                if ring_vertical_angles[i] == new_data[j,3]:
                    tmp.append(new_data[j,:])
            ring_64.append(tmp)
        
        # 16채널
        ring_ds = []
        if ch == 16:
            for i in range(len(ring_64)):
                if i % 4 == 0:
                    ring_ds.append(ring_64[i])
        # 48채널
        elif ch == 48:
            for i in range(len(ring_64)):
                if i % 4 != 0:
                    ring_ds.append(ring_64[i])
                    
        flattened = np.array([item for sublist in ring_ds for item in sublist])
        flattened = np.array(list(chain.from_iterable(ring_ds)))
        #print(flattened[:,0:4])
        return flattened[:,0:3]
    
    def get_empty_coords(self, path):
        voxel_label = np.load(path)
        voxel_label = torch.from_numpy(voxel_label)
        empty_idx = torch.nonzero(voxel_label[0,:,:,:] == 1)

        return empty_idx

    def pad_or_truncate(self, np_array):
        max_points = 50000  # 예시로, 고정할 포인트 수
        if np_array.shape[0] > max_points:
            return np_array[:max_points, :]
        elif np_array.shape[0] < max_points:
            #padding = np.zeros((max_points - np_array.shape[0], np_array.shape[1]), dtype=np_array.dtype)
            padding = np.full((max_points - np_array.shape[0], np_array.shape[1]), np.nan, dtype=np.float32)
            return np.vstack([np_array, padding])
        return np_array

    def __getitem__(self, idx):
        # get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]
        
        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)
            
        # for self-occ mask
        # scene_idx: 폴더
        # sample_idx_in_scene: 폴더 내의 몇번째 scene인지
        scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[idx]
        scene_dir = self.dataset.scenes[scene_idx].directory
        scene_name = os.path.basename(scene_dir)
        mask_idx = self.mask_idx_dict[int(scene_name)]
        
        pc_datum_data, _ = self.dataset.get_point_cloud_from_datum(scene_idx, sample_idx_in_scene, 'lidar')
        pc_filename = self.dataset.get_datum(scene_idx, sample_idx_in_scene, 'lidar').datum.point_cloud.filename
        
        #vx_path = '/dataset/ddad_train_val/occupancy/' + os.path.splitext(os.path.join(os.path.basename(scene_dir), pc_filename))[0] + ".npy"
        #vx_path = '/dataset/ddad_train_val/occupancy/high_resolution_50/' + os.path.basename(scene_dir) + '/point_cloud/LIDAR/'

        point_cloud = self.get_current('point_cloud', 6)
        cur_L_pose = self.get_current('pose', 6).matrix
        prev_L_pose = self.get_context('pose', 6)[0].matrix
        next_L_pose = self.get_context('pose', 6)[1].matrix
        #point_cloud = self.pad_or_truncate(point_cloud)
        #cur_voxel =  np.load(vx_path + str(self.get_current('timestamp', 6)) + ".npy")#self.get_empty_coords(vx_path + str(self.get_current('timestamp', 6)) + ".npy"),
        #prev_voxel = np.load(vx_path + str(self.get_context('timestamp', 6)[0]) + ".npy")#self.get_empty_coords(vx_path + str(self.get_context('timestamp', 6)[0]) + ".npy"),
        #next_voxel = np.load(vx_path + str(self.get_context('timestamp', 6)[1]) + ".npy")#
        
        #sam_masks = {}
        # loop over all cameras
        for cam in range(self.num_cameras):
            filename = self.get_filename_sf(idx, cam)
            
            data = {
                'idx': idx,
                'dataset_idx': self.dataset_idx,
                'sensor_name': self.get_current('datum_name', cam),
                'contexts': contexts,
                'filename': filename,
                'splitname': '%s_%010d' % (self.split, idx),                
                'rgb': self.get_current('rgb', cam),              
                'intrinsics': self.get_current('intrinsics', cam),
                'point_cloud': point_cloud,
                'cur_L_pose' : cur_L_pose,
                'prev_L_pose' : prev_L_pose,
                'next_L_pose' : next_L_pose,
            }
            
            # for lidar loss -> data['sparse_depth'] = self.generate_sparse_depth_map_sf(idx, cam, filename, data['point_cloud'])
            # print(data['rgb']) -> <PIL.Image.Image image mode=RGB size=1936x1216 at 0x7F619585C610>
            data['cur_depth'], data['prev_depth'], data['next_depth'] = self.generate_sparse_depth_map_sf(idx, cam, self.get_current('point_cloud', 6))
            #data['sparse_depth'] = self.generate_sparse_depth_map_sf(idx, cam, self.get_current('point_cloud', 6))
            #data['prev_depth'] = self.generate_sparse_depth_map_sf(idx, cam, self.get_context('point_cloud', 6)[0], 0)
            #data['next_depth'] = self.generate_sparse_depth_map_sf(idx, cam, self.get_context('point_cloud', 6)[1], 1)
            
            #data['voxel_label'] = np.load(vx_path + str(data['pc_timestamp']) + ".npy", allow_pickle=True)
            
            data['prev_pose'] = self.get_context('pose', cam)[0].matrix
            data['next_pose'] = self.get_context('pose', cam)[1].matrix
            
            
            '''
            if cam == 0:
                voxels, coords, _ = voxelize(data['point_cloud'], np.array([1, 1, 1.5]), np.array([-50, -50, -15, 50, 50, 15]), 5, 200000)
                voxel_occ = VoxelRayCasting(coords)
            
            data['voxels'], data['coords'], data['voxel_occ'] = voxels, coords, voxel_occ 
            '''
            
            #data['range_image'], data['range_image_coord'] = self.generate_range_img(data['point_cloud'])
            '''
            # dgp_dataset.py Code
            
            self.depth_type = depth_type
            self.with_depth = depth_type is not None and depth_type is not ''
            self.input_depth_type = input_depth_type
            self.with_input_depth = input_depth_type is not None and input_depth_type is not ''
            
            -> depth_type(input_depth_type)이 '' and None이 아니면 with_depth(with_input_depth)은 True
            -> base_dataset.py 파일을 보면 val에만 depth_type이 True이므로 generate_depth_map_sf 함수는 평가(val, eval)시에만 실행됨!
            -> input_depth_type은 여기에서 있지도 않음 (신경 ㄴㄴ)
            '''
            
            # if depth is returned
            #print("depth ", idx, self.with_depth)
            if self.with_depth:
                data.update({
                    'depth': self.generate_depth_map_sf(idx, cam, filename)
                })
                # print("data['depth']: ", data['depth'].shape) -> (1216, 1936)
            # if depth is returned
            #print("input_depth ", idx, self.with_input_depth)
            if self.with_input_depth:
                data.update({
                    'input_depth': self.generate_depth_map_sf(idx, cam, filename)
                })
            # if pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics': self.get_current('extrinsics', cam).matrix
                })
                data.update({
                    'pose': self.get_current('pose', cam).matrix
                })
                

            # with mask
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, mask_idx, self.cameras[cam])
                })
            # if context is returned
            if self.has_context:
                data.update({
                    'rgb_context': self.get_context('rgb', cam)
                })
                
                data.update({
                    'point_context': self.get_context('point_cloud', 6)
                })
                
            
            '''
            segment_path = '/dataset/dataset/DDAD/gsam_mask/' + filename + '.npz'
            segment_path = segment_path.replace("{}/", "")
            inst_segment = np.load(segment_path, allow_pickle=True)["masks"] # (N_inst, H, W)
            inst_segment = torch.from_numpy(inst_segment).unsqueeze(1) # (N_inst, 1, H, W)
            sam_masks[cam] = inst_segment#.to(device=images.device, dtype=images.dtype)
            #sam_masks.append(inst_segment)
            '''
            sample.append(data)
            

        # apply same data transformations for all sensors
        if self.data_transform:
            # self.data_transform == packnet_sfm/packnet_sfm/datasets/transforms.py의 train_transform 함수
            # train_transform 함수에서 resize_sample(packnet_sfm/packnet_sfm/datasets/augmentations.py) 함수를 호출하는데 여기에서 이미지 사이즈를 줄임 (사이즈 줄일 이미지의 키 지정 필요!)
            sample = [self.data_transform(smp) for smp in sample] # smp['rgb']: <PIL.Image.Image image mode=RGB size=1936x1216 at 0x7F98BAF57700> -> torch.Size([3, 384, 640])
            
            sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]
        
        # stack and align dataset for our trainer
        sample = stack_sample(sample)
        sample = align_dataset(sample, self.scales, contexts)
        #sample['sam_mask'] = sam_masks
        

        return sample