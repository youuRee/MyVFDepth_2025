# Copyright (c) 2023 42dot. All rights reserved.
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import construct_dataset
from network import *
from external.layers import ResnetEncoder

from .base_model import BaseModel
from .geometry import Pose, ViewRendering, BackprojectDepth, Project3D
from .losses import DepthSynLoss, MultiCamLoss, SingleCamLoss

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.io.image import decode_image, read_image
from torchvision.transforms.functional import to_pil_image

from pytorch3d.transforms import axis_angle_to_matrix

from omegaconf import DictConfig
#from .opensf.src.trainer import ModelWrapper

from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R, Slerp

#from .grounded_sam import GroundedSegmentAnything

# Only SAM
#sys.path.append("..")
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

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



_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']#, 'sam_mask']


class VFDepthAlgo(BaseModel):
    """
    Model class for "Self-supervised surround-view depth estimation with volumetric feature fusion"
    """
    def __init__(self, cfg, rank):
        super(VFDepthAlgo, self).__init__(cfg)
        self.rank = rank
        self.read_config(cfg)
        self.prepare_dataset(cfg, rank)
        self.models = self.prepare_model(cfg, rank)   
        self.losses = self.init_losses(cfg, rank)        
        self.view_rendering, self.pose = self.init_geometry(cfg, rank) 

        self.bool_Depth = None
        self.bool_CmpFlow = None
        self.bool_MotMask = None
        self.start_flag = None
        
        #checkpoint = '/workspace/OpenSceneFlow/pretrained/seflow_best.ckpt'
        #self.seflow = ModelWrapper.load_from_checkpoint(checkpoint, eval=True).cuda()
        #for param in self.seflow.parameters():
        #    param.requires_grad = False
        
        #self.unet = UNetFlowInterp().cuda()
        #self.models['refine_net'] = FlowRefiner().cuda()
        self.backproject_depth = BackprojectDepth(self.batch_size, self.height, self.width).to('cuda')
        self.project_3d = Project3D(self.batch_size, self.height, self.width).to('cuda')
        '''
        self.grounded_sam = GroundedSegmentAnything(
                            config_file='/workspace/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', 
                            grounded_checkpoint='/workspace/Grounded-Segment-Anything/groundingdino_swint_ogc.pth', 
                            sam_checkpoint='/workspace/Grounded-Segment-Anything/sam_vit_h_4b8939.pth', 
                            sam_version='vit_h', 
                            device='cuda')
        
        
        SAM
        sam_checkpoint = "/workspace/sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "default"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_mask_generator = SamAutomaticMaskGenerator(sam)
        '''
        #seg_weights = FCN_ResNet50_Weights.DEFAULT
        #self.seg_model = fcn_resnet50(weights=seg_weights).cuda()
        #self.seg_model.eval()
        #self.class_to_idx = {cls: idx for (idx, cls) in enumerate(seg_weights.meta["categories"])}
        
        self.set_optimizer()
        
        if self.pretrain and rank == 0:
            self.load_weights()
    
    # Pretrained  
    '''  
    def load_weights(self, weight_path=None):
        print("Retrained!!!")
        #weight_path=cfg.weight_path
        print(weight_path)
        depth_path = weight_path + 'depth_net.pth'
        pose_path = weight_path + 'pose_net.pth'
        #refine_path = weight_path + 'refine_net.pth'
        optimizer_path = weight_path + 'adam.pth'

        """저장된 가중치와 옵티마이저 상태를 로드하는 메서드"""
        self.models['depth_net'].load_state_dict(torch.load(depth_path))
        self.models['pose_net'].load_state_dict(torch.load(pose_path))
        #self.models['refine_net'].load_state_dict(torch.load(refine_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
    
    
    def save_model(self, epoch):
        """현재 에포크의 모델 가중치와 옵티마이저 상태를 저장하는 메서드"""
        torch.save(self.models['depth_net'].state_dict(), 'depth.pth')
        torch.save(self.models['pose_net'].state_dict(), 'pose.pth')
        torch.save(self.models['refine_net'].state_dict(), 'refine.pth')
        torch.save(self.optimizer.state_dict(), 'adam.pth')
    '''
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_geometry(self, cfg, rank):
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose
        
    def init_losses(self, cfg, rank):
        # config 파일에 따라 loss 계산 달라짐 (ddp는 MultiCamLoss)
        if self.aug_depth:
            loss_model = DepthSynLoss(cfg, rank)
        elif self.spatio_temporal or self.spatio:
            loss_model = MultiCamLoss(cfg, rank)
        else :
            loss_model = SingleCamLoss(cfg, rank)
        return loss_model
        
    def prepare_model(self, cfg, rank):
        models = {}
        models['pose_net'] = self.set_posenet(cfg)        
        models['depth_net'] = self.set_depthnet(cfg)  

        # DDP training
        if self.ddp_enable == True:
            from torch.nn.parallel import DistributedDataParallel as DDP            
            process_group = dist.new_group(list(range(self.world_size)))
            # set ddp configuration
            for k, v in models.items():
                # 학습 파라미터가 존재하지 않는 서브모듈은 DDP로 감싸지 않고, 정상적으로 학습되는 서브모듈만 DDP 적용
                v = v.to(rank)
                if any(p.requires_grad for p in v.parameters()):
                    v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v, process_group)
                    models[k] = DDP(v, device_ids=[rank], broadcast_buffers=True, find_unused_parameters=True)
                else:
                    # 파라미터가 없는 경우 DDP로 감싸면 에러 → 그냥 사용
                    models[k] = v
                '''
                # sync batchnorm
                v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v, process_group)
                # DDP enable
                models[k] = DDP(v, find_unused_parameters=True, device_ids=[rank], broadcast_buffers=True)
                '''
        return models

    def set_motionnet(self, cfg):
        return MotionDecoder(cfg).cuda()

    def set_posenet(self, cfg):
        if self.pose_model =='fusion':
            return FusedPoseNet(cfg).cuda()
        elif self.pose_model =='gt':
            return GTPose(cfg).cuda()
        elif self.pose_model =='pnp':
            return PnPPose(cfg).cuda()
        else:
            return MonoPoseNet(cfg).cuda()    
        
    def set_depthnet(self, cfg):
        if self.depth_model == 'fusion':
            return FusedDepthNet(cfg).cuda()
        else:
            return MonoDepthNet(cfg).cuda()

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')
        
        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            if rank == 0 :
                self.set_val_dataloader(cfg)
                
        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)

    def set_train_dataloader(self, cfg, rank):                 
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)), 
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct train dataset
        train_dataset = construct_dataset(cfg, 'train', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        if self.ddp_enable:
            dataloader_opts['shuffle'] = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, 
                num_replicas = self.world_size,
                rank=rank, 
                shuffle=True
            ) 
            dataloader_opts['sampler'] = self.train_sampler

        self._dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)    
        self.num_total_steps = num_train_samples // (self.batch_size * self.world_size) * self.num_epochs

    def set_val_dataloader(self, cfg):         
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        val_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['val']  = DataLoader(val_dataset, **dataloader_opts)
    
    def set_eval_dataloader(self, cfg):  
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        eval_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)

    def set_optimizer(self):
        parameters_to_train = []
        for v in self.models.values():
            parameters_to_train += list(v.parameters())

        # ✅ motion networks 추가
        #parameters_to_train += list(self.motion_enc.parameters())
        #parameters_to_train += list(self.motion_dec.parameters())
        #parameters_to_train += list(self.motion_mask.parameters())

        self.optimizer = optim.Adam(
        parameters_to_train, 
            self.learning_rate
        )

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            self.scheduler_step_size,
            0.1
        )
    
    def process_batch(self, inputs, rank):
        """
        Pass a minibatch through the network and generate images, depth maps, and losses.
        """
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(rank)   
                    
        outputs = self.estimate_vfdepth(inputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses  


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

    
    def interpolate_extrinsics(self, ext_a, ext_b):
        """
        ext_a, ext_b: [4, 4] extrinsics (T_cam←world)
        weight: interpolation weight (0.5 = halfway)
        Returns: [4, 4] interpolated extrinsic
        """
        # 분해
        R1 = ext_a[:3, :3].cpu().numpy()
        R2 = ext_b[:3, :3].cpu().numpy()
        t1 = ext_a[:3, 3]
        t2 = ext_b[:3, 3]

        # Slerp for rotation
        # 예: 약간만 보간 (중간에 가깝게)
        rot_weight = 0.2 + 0.6 * torch.rand(1).item()  # → [0.2, 0.8] 사이의 값
        key_times = [0, 1]
        key_rots = R.from_matrix([R1, R2])
        slerp = Slerp(key_times, key_rots)
        r_mid = slerp([rot_weight]).as_matrix()[0]

        # Linear interp for translation
        trans_weight = 0.5#0.4 + 0.2 * torch.rand(1).item()  # → [0.4, 0.6] 사이의 값
        t_mid = (1 - trans_weight) * t1 + trans_weight * t2

        # 재구성
        T_mid = torch.eye(4, device=ext_a.device, dtype=ext_a.dtype)
        T_mid[:3, :3] = torch.from_numpy(r_mid).to(ext_a.device, dtype=ext_a.dtype)
        T_mid[:3, 3] = t_mid
        return T_mid
    
    def augment_extrinsics_pairwise(self, ext):
        """
        ext: [B, N_cam, 4, 4]
        return: ext_aug [B, N_cam, 4, 4]
        """
        ref_map = {0: 1, 1: 3, 3: 5, 5: 4, 4: 2, 2: 0}
        B, N, _, _ = ext.shape
        ext_aug = torch.zeros_like(ext)

        for b in range(B):
            for cam in range(N):
                ref_cam = ref_map[cam]
                ext_aug[b, cam] = self.interpolate_extrinsics(ext[b, cam], ext[b, ref_cam])
        return ext_aug

    @torch.no_grad()
    def extract_segment(self, images):
        sam_masks = {}
        for cam in range(6):
            image = images[:, cam, ...][0]#.permute(1, 2, 0).detach().cpu().numpy()
            
            # SAM 등이 uint8 RGB를 기대하는 경우:
            #if image.dtype != np.uint8:
            #    image_for_mask = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            #else:
            #    image_for_mask = image
            # RGB라고 가정. 만약 원본이 BGR이면 cvtColor로 RGB로 바꾸세요.
            # image_for_mask = cv2.cvtColor(image_for_mask, cv2.COLOR_BGR2RGB)
            
            # (num_segment, 1, h, w)
            segs, _, _ = self.grounded_sam.predict(input_image=image, 
                         text_prompt='car . bus . truck . cyclist . person . tree . pole . fence . building . traffic light . traffic sign . sky', )
            '''
            SAM
            masks = self.sam_mask_generator.generate(image_for_mask)  # list of dicts
            # torch 텐서로 변환 (각각 (1,H,W))
            segs = [torch.from_numpy(d['segmentation'].astype('float32')[None, ...]) for d in masks]
            segs = torch.stack(segs, dim=0) if len(segs) > 0 else torch.zeros((0,1,H,W))
            '''
            sam_masks[cam] = segs.to(device=images.device, dtype=images.dtype)
            
        #seg_masks = torch.cat(seg_masks, dim=0) # (6,79,1,h,w) -> (num_cam, num_segment, 1, h, w)
        return sam_masks
    
    '''
    def load_segment(self, inputs):
        sam_masks = {}
        for cam in range(6):
            filename = 
            segment_path = '/dataset/dataset/DDAD/gsam_mask/' + filename + '.npz'
            segment_path = segment_path.replace("{}/", "")
            inst_segment = np.load(segment_path, allow_pickle=True)["masks"] # (N_instance, H, W)
    '''
    def estimate_vfdepth(self, inputs):
        """
        This function sets dataloader for validation in training.
        """          
        # pre-calculate inverse of the extrinsic matrix        
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        inputs['extrinsics_aug'] = self.augment_extrinsics_pairwise(inputs['extrinsics'])
        #inputs['sam_mask'] = self.extract_segment(inputs[('color', 0, 0)]) #-> 학습할때만
        
        #print(inputs.keys())
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}


        pose_pred = self.predict_pose(inputs)                
        depth_feats = self.predict_depth(inputs)

        for cam in range(self.num_cams):       
            outputs[('cam', cam)].update(pose_pred[('cam', cam)])              
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])
            #outputs[('cam', cam)].update(depth_feats[('cam', cam)])

        if self.syn_visualize:
            outputs['disp_vis'] = depth_feats['disp_vis']
        
        
        self.compute_depth_maps(inputs, outputs)
        #self.predict_sceneflow(inputs, outputs)
        
        return outputs
            
    def generate_2dflow(self, inputs, outputs, cam, frame_id, pc, pose, flow):
        f_i = frame_id[0]
        N = pc.shape[0]
        ones = torch.ones((N, 1), device=pc.device)
        X_L_homo = torch.cat([pc[:, :3], ones], dim=1) # 동차좌표 추가 (N, 4)
        X_W = (pose @ X_L_homo.T).T # 월드 좌표계로 변환 (N, 4)
        
        p_wc = inputs[("pose", f_i)][:,cam,:,:][0]
        p_cw = p_wc.inverse()
        k = inputs[('K', 0)][:,cam,:,:][0]
        Xc = (p_cw @ X_W.T).T # 카메라 좌표계로 변환 (N, 4)
        
        # 이미지 평면 투영 (N, 2)
        cam_coords = Xc.T  # (3, N)
        pix_coords = k @ cam_coords  # (3, N)
        ori_proj = pix_coords[:2, :] / pix_coords[2:3, :]  # (2, N)
        
        flow_pc = pc + flow
        X_L_homo = torch.cat([flow_pc[:, :3], ones], dim=1) # 동차좌표 추가 (N, 4)
        X_W = (pose @ X_L_homo.T).T # 월드 좌표계로 변환 (N, 4)
        Xc = (p_cw @ X_W.T).T # 카메라 좌표계로 변환 (N, 4)
        cam_coords = Xc.T  # (3, N)

        pix_coords = k @ cam_coords  # (3, N)
        flow_proj = pix_coords[:2, :] / pix_coords[2:3, :]  # (2, N)
        
        flow_2d = flow_proj - ori_proj
        
        # Assume torch.Tensor inputs
        H, W = 384, 640
        device = flow_2d.device

        # Initialize flow and count maps
        flow_map = torch.zeros((2, H, W), dtype=torch.float32, device=device)
        count_map = torch.zeros((H, W), dtype=torch.float32, device=device)

        # Convert to pixel coordinates
        xy = ori_proj.long()  # shape (2, N)
        x, y = xy[0], xy[1]
        z = cam_coords[2]

        # valid index mask
        z_valid_mask = z > 0
        valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        valid_mask = valid_mask & z_valid_mask

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        u_valid = flow_2d[0][valid_mask]
        v_valid = flow_2d[1][valid_mask]

        # scatter_add: add flow to correct pixel
        # 채널 인덱스 생성
        c0 = torch.zeros_like(x_valid)
        c1 = torch.ones_like(x_valid)
        flow_map.index_put_((c0, y_valid, x_valid), u_valid, accumulate=True)
        flow_map.index_put_((c1, y_valid, x_valid), v_valid, accumulate=True)
        count_map.index_put_((y_valid, x_valid), torch.ones_like(u_valid), accumulate=True)

        # Avoid division by zero
        count_map = count_map + 1e-6

        # Normalize
        flow_map[0] /= count_map
        flow_map[1] /= count_map
        
        '''
        # Permute to (H, W, 2) for visualization
        flow_map_np = flow_map.permute(1, 2, 0).detach().cpu().numpy()

        # Visualize
        flow_vis = flow_to_color(flow_map_np)

        plt.figure(figsize=(20, 10))
        plt.imshow(flow_vis)
        plt.title(f"CAM{cam}, frame_id={frame_id[1]}")
        plt.axis('off')
        plt.savefig('flow_map.png')
        
        # 1. Flow magnitude 계산
        flow_mag = torch.norm(flow_map, dim=0)  # (H, W)

        # 2. 임계값 기준 설정
        mean_mag = flow_mag[flow_mag > 0].mean()
        threshold = mean_mag * 1.5  # 필요 시 조절 가능

        # 3. 동적 객체 마스크 생성 (1: dynamic, 0: static)
        motion_mask = (flow_mag > threshold).float()  # (H, W)

        # Optional: 시각화용 (gray image)
        motion_mask_np = motion_mask.detach().cpu().numpy()
        plt.figure(figsize=(20, 10))
        plt.imshow(motion_mask_np, cmap='gray')
        plt.title(f"CAM{cam}, frame_id={frame_id[1]}")
        plt.axis('off')
        plt.savefig('motion_mask.png')
        '''
        
        return flow_map
    
    def predict_sceneflow(self, inputs, outputs):

        """ Predict egomotion (in a form of rotation and translation)
        """
        for cam in range(self.num_cams): 
            for frame_id in [[0, -1], [0, 1]]:
                pc0 = inputs[('lidar', frame_id[0])]
                pc1 = inputs[('lidar', frame_id[1])]
                pose0 = inputs[('L_pose', frame_id[0])][:,0,:,:]#[0][0].unsqueeze(0)
                pose1 = inputs[('L_pose', frame_id[1])][:,0,:,:]#[0][0].unsqueeze(0)
                
                inp = {
                    "pc0": pc0,
                    "pc1": pc1,
                    "pose0": pose0,
                    "pose1": pose1
                }
                out = self.seflow.model(inp)
                
                batch_id = 0
                pose_flows = out['pose_flow']
                valid_from_pc2res = out['pc0_valid_point_idxes'][batch_id]
                pose_flow = pose_flows[batch_id][valid_from_pc2res]
                final_flow = pose_flow.clone() + out['flow'][batch_id]
                
                sparse_flow = self.generate_2dflow(inputs, outputs, cam, frame_id, out['pc0_points_lst'][batch_id], pose0[0], final_flow).unsqueeze(0)
                outputs[('cam', cam)].update({('sparse_flow', frame_id[1], 0) :   sparse_flow})
                
                # sparse to dense flow
                source_scale = 0
                scale = 0
                
                ref_depth = outputs[('cam', cam)][('depth', scale)]
                T = outputs[('cam', cam)][('cam_T_cam', 0, frame_id[1])]
                K = inputs[('K', source_scale)][:, cam, ...]

                cam_points = self.backproject_depth(ref_depth, inputs[('inv_K', source_scale)][:, cam, ...])
                outputs[('cam', cam)][('cam_points', 0, scale)] = cam_points
                
                _, _, ego_flow = self.project_3d(cam_points, K, T)  # (B, H, W, 2), (B, 3, H*W)
                ego_flow = ego_flow.permute(0, 3, 1, 2)  # (B, 2, H, W)
                outputs[('cam', cam)][('ego_flow', frame_id[1], scale)] = ego_flow
                
                '''
                if self.start_flag and not self.pretrain:
                    # 학습된 refine_net이 없는 경우
                    valid_mask = (sparse_flow != 0).float()
                    dense_flow = sparse_flow * valid_mask + ego_flow * (1 - valid_mask)  # (B, 2, H, W)
                
                else:
                    valid_mask = (sparse_flow != 0).float()[:,0,:,:].unsqueeze(0)
                    dense_input = torch.cat([sparse_flow, ego_flow, valid_mask], dim=1)  # (B, 5, H, W)
                    dense_flow = self.models['refine_net'](dense_input)  # (B, 2, H, W)
                
                
                prediction = self.seg_model(inputs[('color', 0, 0)][:,cam,:,:])["out"]
                normalized_masks = prediction.softmax(dim=1)
                seg_mask = normalized_masks[0, self.class_to_idx["person"]] + normalized_masks[0, self.class_to_idx["car"]] + normalized_masks[0, self.class_to_idx["bus"]] + normalized_masks[0, self.class_to_idx["motorbike"]] + normalized_masks[0, self.class_to_idx["bicycle"]]
                
                seg_mask = (seg_mask > 0.5)#.float().unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
                flow_mask = (sparse_flow != 0)[0][0]

                np_mask = ((flow_mask & seg_mask) + ~seg_mask).detach().cpu().numpy()
                distances, indices = distance_transform_edt(~np_mask, return_indices=True) # (2, 1216, 1936)

                filled_flow = sparse_flow.clone()
                device = sparse_flow.device
                
                indices_row = torch.from_numpy(indices[0]).to(device).long()  # (H, W)
                indices_col = torch.from_numpy(indices[1]).to(device).long()  # (H, W)

                # fill in torch
                for c in range(sparse_flow.shape[1]):
                    channel_flow = filled_flow[0, c]
                    nearest_values = channel_flow[indices_row, indices_col]
                    channel_flow[~torch.from_numpy(np_mask).to(device)] = nearest_values[~torch.from_numpy(np_mask).to(device)]

                valid_mask = (filled_flow != 0).float() #seg_mask.float().unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
                dense_flow = valid_mask * filled_flow + (1 - valid_mask) * ego_flow
                '''
                
                pooling_flow = F.avg_pool2d(sparse_flow, kernel_size=5, stride=1, padding=5//2)
                
                valid_mask = (pooling_flow != 0).float()
                dense_flow = pooling_flow * valid_mask + ego_flow * (1 - valid_mask)  # (B, 2, H, W)
                
                outputs[('cam', cam)].update({('dense_flow', frame_id[1], 0) :   dense_flow})
                
                #valid_mask = (flow_map_2d != 0).float()[:,0,:,:].unsqueeze(0)
                #dense_flow = self.unet(torch.cat([flow_map_2d, valid_mask], dim=1))
                
                '''
                # Permute to (H, W, 2) for visualization
                flow_map_np = dense_flow[0].permute(1, 2, 0).detach().cpu().numpy()

                # Visualize
                flow_vis = flow_to_color(flow_map_np)
                plt.figure(figsize=(20, 10))
                plt.imshow(flow_vis)
                plt.title(f"CAM{cam}, frame_id={frame_id[1]}")
                plt.axis('off')
                plt.savefig('dense_flow_map.png')
                '''
                
    
    
    def predict_motion_feat(self, inputs, outputs, cam):

        """ Predict egomotion (in a form of rotation and translation)
        """

        # 여기서 4DMOS 돌리기
        # 결과 피쳐를 dynamic_util의 Project3D처럼 pix_coord를 얻어서 이미지 형태로 만들기
        curr_pts = inputs[('lidar', 0)][0]
        from_pose = inputs[('L_pose', -1)][0][0]
        to_pose = inputs[('L_pose', 0)][0][0]
        past2curr_pts = self.transform_point_cloud(inputs[('lidar', -1)][0], from_pose, to_pose)
        
        curr_pts = self.timestamp_tensor(curr_pts, 0.1)
        past2curr_pts = self.timestamp_tensor(past2curr_pts, 0.0)
        
        accum_point_clouds = [past2curr_pts, curr_pts]
        #out = self.models['mos_net'](accum_point_clouds)
        out = self.mosnet(accum_point_clouds)
        
        b = 1
        t = 0.1
        coords = out.coordinates_at(b) # point의 voxel 좌표
        logits = out.features_at(b) # point의 특징 값 
        
        mask = coords[:, -1].isclose(torch.tensor(t)) # 특정 시점 t에 해당하는 포인트만 골라내기 
        masked_logits = logits[mask]
        #masked_logits[:, self.models['mos_net'].ignore_index] = -float("inf")
        masked_logits[:, self.mosnet.ignore_index] = -float("inf")
        features = F.softmax(masked_logits, dim=1)
        
        N = coords.shape[0]
        ones = torch.ones((N, 1), device=coords.device)
        X_L_homo = torch.cat([coords[:, :3], ones], dim=1) # 동차좌표 추가 (N, 4)
        X_W = (to_pose @ X_L_homo.T).T # 월드 좌표계로 변환 (N, 4)
        
        p_wc = inputs[("pose", 0)][:,cam,:,:][0]
        p_cw = p_wc.inverse()
        k = inputs[('K', 0)][:,cam,:,:][0]
        Xc = (p_cw @ X_W.T).T # 카메라 좌표계로 변환 (N, 4)
        
        # 이미지 평면 투영 (N, 2)
        cam_coords = Xc.T  # (3, N)
        pix_coords = k @ cam_coords  # (3, N)
        pix_coords = pix_coords[:2, :] / pix_coords[2:3, :]  # (2, N)
        # pix_coords → (N, 2)
        pix_coords = pix_coords.T  # (N, 2), (x, y)
        
        B = 1  # batch size
        C = features.shape[1]  # feature channels
        H, W = 384, 640  # 이미지 높이, 너비
        
        x_pix = pix_coords[:, 0].round().long()
        y_pix = pix_coords[:, 1].round().long()

        # 이미지 범위 제한 (clamp)
        x_pix = x_pix.clamp(0, W - 1)
        y_pix = y_pix.clamp(0, H - 1)
        
        mos_feat = torch.zeros((B, C, H, W), device=features.device)
        for c in range(C):
            # 하나의 채널씩 scatter_add (batch dim은 0이니까 무시)
            mos_feat[0, c].index_put_(
                (y_pix, x_pix),  # (행, 열)
                features[:, c],
                accumulate=True  # 같은 위치에 여러 값이 오면 누적
            )

        #plt.imshow(mos_feat[0].permute(1, 2, 0).detach().cpu().numpy(), 'hot')
        #plt.axis('off')
        #plt.savefig("mos_feat.png")


        #motion_inputs = {f_i: self.camli_conv2d(torch.cat([inputs["color_aug", f_i, 0][:,cam,:,:], inputs["gt_depth", f_i][:,cam,:,:]], dim=1)) for f_i in self.frame_ids}
        motion_inputs = {f_i: inputs["color_aug", f_i, 0][:,cam,:,:] for f_i in self.frame_ids}
        
        for f_gap in set([abs(f_i) for f_i in self.frame_ids[1:]]):
            f_prev, f_next = -1 * f_gap, f_gap

            # Keep order and center frame at zero
            motion_input = torch.cat([motion_inputs[f_prev], motion_inputs[0], motion_inputs[f_next], mos_feat], 1)
            motion_feats = self.models['motion_enc'](motion_input)
            
            # motion_input: 3시점의 이미지만 혹은 mos_feat도 추가 (motion_dec), mos_feat만 (motion_mask)
            # motion_feats: 이미지, mos_feat 둘 다 넣어서 Encoding 한 피쳐 (motion_dec, motion_mask)
            # motion_enc: ResNet 말고 다른 Encoding 방법 (ex. VFNet)
            # mos_feat: 몇개의 시점 데이터를 쓸 것인지... (t만 or t-1, t, t+1 다)
            outputs[('cam', cam)].update({
                ('motion_feats', 0, f_gap) :   [motion_input] + motion_feats,
            })

    
    def predict_motions(self, inputs, outputs):

        """ Predict independent object motion 
        """

        if not self.bool_CmpFlow and not self.bool_MotMask:
            return  # early terminate since no need for output

        for cam in range(self.num_cams): 
            self.predict_motion_feat(inputs, outputs, cam)

            for f_gap in set([abs(f_i) for f_i in self.frame_ids[1:]]):
                f_prev, f_next = -1 * f_gap, f_gap

                motion_input = outputs[('cam', cam)][('motion_feats', 0, f_gap)]

                # subtraction since order is ignored during, obtain its mean
                ego_translation = (outputs[('cam', cam)][('translation', 0, f_prev)].detach() - outputs[('cam', cam)][('translation', 0, f_next)].detach()) / 2
                ego_axisangle = (outputs[('cam', cam)][('axisangle', 0, f_prev)].detach() - outputs[('cam', cam)][('axisangle', 0, f_next)].detach()) / 2
                ego_motion = torch.cat((ego_translation, ego_axisangle), -1).permute(0,2,1).unsqueeze(3)
                

                if self.bool_CmpFlow:
                    motion_out = self.models['motion_dec'](motion_input, ego_motion)

                    # full motion predictions need to be inverted for finding a point back in time
                    # always using frame 0 as reference, so it is omitted -> (name, f_i, scale)
                    outputs[('cam', cam)].update({(k[0], f_prev, k[1]) : -1 * v for k,v in motion_out.items()}) 
                    outputs[('cam', cam)].update({(k[0], f_next, k[1]) :  1 * v for k,v in motion_out.items()}) 
                
                
                if self.bool_MotMask:
                    motion_prob = self.models['motion_mask'](motion_input, ego_motion)

                    # motion probabilities just need to be duplicated
                    # always using frame 0 as reference, so it is omitted -> (name, f_i, scale)
                    outputs[('cam', cam)].update({(k[0], f_prev, k[1]) : v for k,v in motion_prob.items()})
                    outputs[('cam', cam)].update({(k[0], f_next, k[1]) : v for k,v in motion_prob.items()})
                
        #print("AFter motion\n", outputs[('cam', 0)].keys())

    def predict_pose(self, inputs):      
        """
        This function predicts poses.
        """          
        net = None
        if (self.mode != 'train') and self.ddp_enable:
            net = self.models['pose_net'].module
        else:
            net = self.models['pose_net']
        
        pose = self.pose.compute_pose(net, inputs)
        return pose

    def predict_depth(self, inputs):
        """
        This function predicts disparity maps.
        """                  
        net = None
        if (self.mode != 'train') and self.ddp_enable: 
            net = self.models['depth_net'].module
        else:
            net = self.models['depth_net']

        if self.depth_model == 'fusion':
            depth_feats = net(inputs)
        else:         
            depth_feats = {}
            for cam in range(self.num_cams):
                input_depth = inputs[('color_aug', 0, 0)][:, cam, ...]
                depth_feats[('cam', cam)] = net(input_depth)
        return depth_feats
    
    def compute_depth_maps(self, inputs, outputs):     
        """
        This function computes depth map for each viewpoint.
        """ 
        # outputs(즉, 모델 통과해서 나온 예측 값)을 to_depth 함수를 통해 depth map으로 만들어주기
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...]
            for scale in self.scales:
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', scale)] = self.to_depth(disp, ref_K)
                if self.aug_depth:
                    disp = outputs[('cam', cam)][('disp', scale, 'aug')]
                    outputs[('cam', cam)][('depth', scale, 'aug')] = self.to_depth(disp, ref_K)
    
                '''
                for frame_id in self.frame_ids[1:]:  
                    disp = outputs[('cam', cam)][('disp', scale)]
                    baseline = torch.norm(outputs[('cam',cam)][('cam_T_cam', 0, frame_id)][0,:3,3])
                    outputs[('cam', cam)][('depth', scale, 0, frame_id)] = self.to_depth(disp, ref_K, baseline)
                    #print(frame_id, " ", outputs[('cam', cam)].keys())
                outputs[('cam', cam)][('depth', scale)] = (outputs[('cam', cam)][('depth', scale, 0, -1)] + outputs[('cam', cam)][('depth', scale, 0, 1)]) / 2

                if self.aug_depth:
                    for frame_id in self.frame_ids[1:]:  
                        disp = outputs[('cam', cam)][('disp', scale)]
                        baseline = torch.norm(outputs[('cam',cam)][('cam_T_cam', 0, frame_id)][0,:3,3])
                        disp = outputs[('cam', cam)][('disp', scale, 'aug')]
                        outputs[('cam', cam)][('depth', scale, 'aug', 0, frame_id)] = self.to_depth(disp, ref_K, baseline)
                    outputs[('cam', cam)][('depth', scale, 'aug')] = (outputs[('cam', cam)][('depth', scale, 'aug', 0, -1)] + outputs[('cam', cam)][('depth', scale, 'aug', 0, 1)]) / 2
                '''
    #def sparse_depth_map(self, inputs, outputs):
    
    def to_depth(self, disp_in, K_in):        
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale
    
    def compute_losses(self, inputs, outputs):
        """
        This function computes losses.
        """
        # lidar loss: inputs['point_clud']로 sparse depth map 만들고 output이랑 depth loss 구하기
        losses = 0
        loss_fn = defaultdict(list)
        loss_mean = defaultdict(float)
        '''
        loss_dict(카메라 하나 당 loss):  dict_keys(['reproj_loss', 'spatio_loss', 'spatio_tempo_loss', 'lidar_loss', 'smooth', 'depth/mean', 'depth/max', 'depth/min'])
        
        loss_fn(카메라 별 loss):  dict_keys(['reproj_loss', 'spatio_loss', 'spatio_tempo_loss', 'lidar_loss', 'smooth', 'depth/mean', 'depth/max', 'depth/min', 'pose/tx', 'pose/ty', 'pose/tz'])
        '''
        # generate image and compute loss per cameara
        self.losses.bool_Depth = self.bool_Depth
        #self.losses.bool_CmpFlow = self.bool_CmpFlow
        #self.losses.bool_MotMask = self.bool_MotMask
        #rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)

        for cam in range(self.num_cams):
            self.pred_cam_imgs(inputs, outputs, cam)
            cam_loss, loss_dict = self.losses(inputs, outputs, cam) # init_losses에서 초기화
            losses += cam_loss  
            for k, v in loss_dict.items():
                loss_fn[k].append(v)

        losses /= self.num_cams
        
        for k in loss_fn.keys():
            loss_mean[k] = sum(loss_fn[k]) / float(len(loss_fn[k]))
            #print(k, ': ', loss_mean[k])
          
        loss_mean['total_loss'] = losses        
        return loss_mean

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """                  
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        
        #self.view_rendering.bool_CmpFlow = self.bool_CmpFlow
        #self.view_rendering.bool_MotMask = self.bool_MotMask
        self.view_rendering(inputs, outputs, cam, rel_pose_dict)  