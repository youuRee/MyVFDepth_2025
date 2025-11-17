# Copyright (c) 2023 42dot. All rights reserved.
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as pil

from tensorboardX import SummaryWriter

from .visualize import colormap
from .misc import pretty_ts, cal_depth_error

from .new_metric import evaluate_instance_mean, evaluate_occupancy_iou

def set_tb_title(*args):
    """
    This function sets title for tensorboard plot.
    """    
    title = ''
    for i, s in enumerate(args):
        if not i%2: title += '/'
        s = s if isinstance(s, str) else str(s)
        title += s
    return title[1:]
    

def resize_for_tb(image):
    """
    This function resizes images for tensorboard plot.
    """     
    h, w = image.size()[-2:]
    return F.interpolate(image, [h//2, w//2], mode='bilinear', align_corners=True) 
    

def plot_tb(writer, step, img, title, j=0):
    """
    This function plots images on tensotboard.
    """     
    img_resized = resize_for_tb(img)    
    writer.add_image(title, img_resized[j].data, step)


def plot_norm_tb(writer, step, img, title, j=0):
    """
    This function plots normalized images on tensotboard.
    """     
    img_resized = torch.clamp(resize_for_tb(img), 0., 1.)
    writer.add_image(title, img_resized[j].data, step)


def plot_disp_tb(writer, step, disp, title, j=0):
    """
    This function plots disparity maps on tensotboard.
    """  
    disp_resized = resize_for_tb(disp).float()
    disp_resized = colormap(disp_resized[j, 0])
    writer.add_image(title, disp_resized, step)    

    
class Logger:
    """
    Logger class to monitor training
    """
    def __init__(self, cfg, use_tb=True):
        self.read_config(cfg)
        os.makedirs(self.log_path, exist_ok=True)
        
        if use_tb: 
            self.init_tb()
            
        if self.eval_visualize:
            self.init_vis()

        self._metric_names = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']
        self.txt_path = '/workspace/MyVFDepth_2025/D_LongBinFusion_Large/log_results.txt'

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
        
    def init_tb(self):
        self.writers = {}
        for mode in ['train', 'val']:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        
    def close_tb(self):
        for mode in ['train', 'val']:
            self.writers[mode].close()

    def init_vis(self):
        vis_path = os.path.join(self.log_path, 'vis_results')
        os.makedirs(vis_path, exist_ok=True)
        
        self.cam_paths = []
        for cam_id in range(self.num_cams):
            cam_path = os.path.join(vis_path, f'cam{cam_id:d}')
            os.makedirs(cam_path, exist_ok=True)
            self.cam_paths.append(cam_path)
            
            
        if self.syn_visualize:
            self.syn_path = os.path.join(self.log_path, 'syn_results')
            os.makedirs(self.syn_path, exist_ok=True)
            
    def get_metric_names(self):
        return self._metric_names
    
    def update(self, mode, epoch, world_size, batch_idx, step, start_time, before_op_time, inputs, outputs, losses):
        """
        Display logs with respect to the log frequency
        """    
        # iteration duration
        duration = time.time() - before_op_time

        if self.is_checkpoint(step):
            self.log_time(epoch, batch_idx * world_size, duration, losses, start_time)
            self.log_tb(mode, inputs, outputs, losses, step)
                
    def is_checkpoint(self, step):
        """ 
        Log less frequently after the early phase steps
        """
        early_phase = (step % self.log_frequency == 0) and (step < self.early_phase)
        late_phase = step % self.late_log_frequency == 0
        return (early_phase or late_phase)

    def log_time(self, epoch, batch_idx, duration, loss, start_time):
        """
        This function prints epoch, iteration, duration, loss and spent time.
        """
        rep_loss = loss['total_loss'].item()
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - start_time

        # 로그 메시지 생성
        log_message = f'epoch: {epoch:2d} | batch: {batch_idx:6d} |' + \
                    f'examples/s: {samples_per_sec:5.1f} | loss: {rep_loss:.3f} | time elapsed: {pretty_ts(time_sofar)}'

        # 터미널 출력
        print(log_message)

        # .txt 파일에 저장 (이어쓰기 모드)
        with open(self.txt_path, "a") as f:
            f.write(log_message + "\n")  # 줄바꿈 포함해 저장

        #print(f'epoch: {epoch:2d} | batch: {batch_idx:6d} |' + \
        #      f'examples/s: {samples_per_sec:5.1f} | loss: {rep_loss:.3f} | time elapsed: {pretty_ts(time_sofar)}')
        
    def log_tb(self, mode, inputs, outputs, losses, step):
        """
        This function logs outputs for monitoring using tensorboard.
        """
        #print(f"          | pose_loss = {outputs['pose_loss']:.4f} (pose/tx: {losses['pose/tx'].item():.4f} | pose/ty: {losses['pose/ty'].item():.4f} | pose/tz: {losses['pose/tz'].item():.4f})")
        #print(f"          | flow_loss = {losses['flow_loss']:.4f}")
        writer = self.writers[mode]
        # loss
        for l, v in losses.items():
            writer.add_scalar(f'{l}', v, step)
        
        scale = 0 # plot the maximum scale
        for cam_id in range(self.num_cams):
            target_view = outputs[('cam', cam_id)]
            
            plot_tb(writer, step, inputs[('color', 0, scale)][:, cam_id, ...], set_tb_title('cam', cam_id)) # frame_id 0            
            plot_disp_tb(writer, step, target_view[('disp', scale)], set_tb_title('cam', cam_id, 'disp')) # disparity
            #plot_tb(writer, step, target_view[('reproj_loss', scale)], set_tb_title('cam', cam_id, 'reproj')) # reprojection image
            #plot_tb(writer, step, target_view[('reproj_mask', scale)], set_tb_title('cam', cam_id, 'reproj_mask')) # reprojection mask
            plot_tb(writer,  step, inputs['mask'][:, cam_id, ...], set_tb_title('cam', cam_id, 'self_occ_mask'))
    
            if self.spatio:
                plot_norm_tb(writer, step, target_view[('overlap', 0, scale)], set_tb_title('cam', cam_id, 'sp'))
                plot_tb(writer, step, target_view[('overlap_mask', 0, scale)], set_tb_title('cam', cam_id, 'sp_mask'))
                
            if self.spatio_temporal:
                for frame_id in self.frame_ids:
                    if frame_id == 0:
                        continue
                    plot_norm_tb(writer, step, target_view[('color', frame_id, scale)], set_tb_title('cam', cam_id, 'pred_', frame_id))                      
                    plot_norm_tb(writer, step, target_view[('overlap', frame_id, scale)], set_tb_title('cam', cam_id, 'sp_tm_', frame_id))
                    plot_tb(writer, step, target_view[('overlap_mask', frame_id, scale)], set_tb_title('cam', cam_id, 'sp_tm_mask_', frame_id))
                    
            if self.aug_depth:
                plot_disp_tb(writer, step, target_view[('disp', scale, 'aug')], set_tb_title('view_aug', cam_id))                

    def log_result(self, inputs, outputs, idx, syn_visualize=False):
        """
        This function logs outputs for visualization.
        """        
        scale = 0
        for cam_id in range(self.num_cams):
            target_view = outputs[('cam', cam_id)]
            disps = target_view['disp', scale]
            for jdx, disp in enumerate(disps):       
                disp = colormap(disp)[0,...].transpose(1,2,0)
                disp = pil.fromarray((disp * 255).astype(np.uint8))
                cur_idx = idx*self.batch_size + jdx 
                disp.save(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_disp.jpg'))
            
        if syn_visualize:    
            syn_disps = outputs['disp_vis']
            for kdx, syn_disp in enumerate(syn_disps):
                syn_disp = colormap(syn_disp)[0,...].transpose(1,2,0)
                syn_disp = pil.fromarray((syn_disp * 255).astype(np.uint8))
                syn_disp.save(os.path.join(self.syn_path, f'{kdx:03d}_syndisp.jpg'))
    
    def compute_depth_losses(self, inputs, outputs, syn_depth, vis_scale=False):
        """
        This function computes depth metrics, to allow monitoring of training process on validation dataset.
        """
        min_eval_depth = self.eval_min_depth
        max_eval_depth = self.eval_max_depth
        
        med_scale = []
        
        error_metric_dict = defaultdict(float)
        error_median_dict = defaultdict(float)
        
        for cam in range(self.num_cams):
            target_view = outputs['cam', cam]

            if syn_depth:
                depth_gt = target_view[('aug_lidar_depth', 0)]
                _, _, h, w = depth_gt.shape
                depth_pred = target_view[('depth', 0, 'aug')].to(depth_gt.device)
            else:
                depth_gt = inputs['depth'][:, cam, ...]
                _, _, h, w = depth_gt.shape
                depth_pred = target_view[('depth', 0)].to(depth_gt.device)
            
            depth_pred = torch.clamp(F.interpolate(
                        depth_pred, [h, w], mode='bilinear', align_corners=False), 
                         min_eval_depth, max_eval_depth)
            depth_pred = depth_pred.detach()

            mask = (depth_gt > min_eval_depth) * (depth_gt < max_eval_depth) * inputs['mask'][:, cam, ...]
            mask = mask.bool()
            
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            
            # calculate median scale
            scale_val = torch.median(depth_gt) / torch.median(depth_pred)
            med_scale.append(round(scale_val.cpu().numpy().item(), 2))
                            
            depth_pred_metric = torch.clamp(depth_pred, min=min_eval_depth, max=max_eval_depth)
            depth_errors_metric = cal_depth_error(depth_pred_metric, depth_gt)
            
            depth_pred_median = torch.clamp(depth_pred * scale_val, min=min_eval_depth, max=max_eval_depth)
            depth_errors_median = cal_depth_error(depth_pred_median, depth_gt)
            
            for i in range(len(depth_errors_metric)):
                key = self._metric_names[i]
                error_metric_dict[key] += depth_errors_metric[i]
                error_median_dict[key] += depth_errors_median[i]

        if vis_scale==True:
            # print median scale
            print(f'          | median scale = {med_scale}')
                
        for key in error_metric_dict.keys():
            error_metric_dict[key] = error_metric_dict[key].cpu().numpy() / self.num_cams
            error_median_dict[key] = error_median_dict[key].cpu().numpy() / self.num_cams
            
        return error_metric_dict, error_median_dict 
    
    
    def compute_invalid_depth_losses(self, inputs, outputs, syn_depth, vis_scale=False):
        """
        This function computes depth metrics, to allow monitoring of training process on validation dataset.
        """
        min_eval_depth = self.eval_min_depth
        max_eval_depth = self.eval_max_depth
        
        error_inst_dict = defaultdict(float)
        iou_bev_dict = defaultdict(float)
        
        for cam in range(self.num_cams):
            target_view = outputs['cam', cam]

            if syn_depth:
                depth_gt = target_view[('aug_lidar_depth', 0)]
                _, _, h, w = depth_gt.shape
                depth_pred = target_view[('depth', 0, 'aug')].to(depth_gt.device)
            else:
                depth_gt = inputs['depth'][:, cam, ...]
                _, _, h, w = depth_gt.shape
                depth_pred = target_view[('depth', 0)].to(depth_gt.device)
            
            depth_pred = torch.clamp(F.interpolate(
                        depth_pred, [h, w], mode='bilinear', align_corners=False), 
                         min_eval_depth, max_eval_depth)
            depth_pred = depth_pred.detach()
                            
            depth_pred_metric = torch.clamp(depth_pred, min=min_eval_depth, max=max_eval_depth)
            
            invK = inputs[('inv_K', 0)][:,cam,:,:]
            seg_mask = inputs['sam_mask'][cam][0] # (N, 1, H, W)
            seg_mask = seg_mask.to(device=depth_gt.device, dtype=depth_gt.dtype)
            
            depth_errors_inst = evaluate_instance_mean(depth_gt, depth_pred_metric, seg_mask)
            depth_iou_bev = evaluate_occupancy_iou(depth_pred_metric, depth_gt, invK)
            
            for key in depth_errors_inst:
                error_inst_dict[key] += depth_errors_inst[key]
            
            for key in depth_iou_bev:
                iou_bev_dict[key] += depth_iou_bev[key]
                
        for key in error_inst_dict.keys():
            error_inst_dict[key] = error_inst_dict[key] / self.num_cams
        
        for key in iou_bev_dict.keys():     
            iou_bev_dict[key] = iou_bev_dict[key] / self.num_cams
        
        return error_inst_dict, iou_bev_dict
                
    def compute_depth_losses_eval(self, inputs, outputs, vis_scale=False):
        """
        This function computes depth metrics, to allow monitoring of training process on validation dataset.
        """
        min_eval_depth = self.eval_min_depth
        max_eval_depth = self.eval_max_depth
        
        med_scale_1 = []
        med_scale_2 = []
        
        error_metric_dict_1 = defaultdict(float)
        error_median_dict_1 = defaultdict(float)
        error_metric_dict_2 = defaultdict(float)
        error_median_dict_2 = defaultdict(float)
        
        for cam in range(self.num_cams):
            target_view = outputs['cam', cam]
            
            depth_gt = inputs['depth'][:, cam, ...]
            _, _, h, w = depth_gt.shape
            depth_pred = target_view[('depth', 0)].to(depth_gt.device)
                
            depth_pred = torch.clamp(F.interpolate(
                         depth_pred, [h, w], mode='bilinear', align_corners=False), 
                         min_eval_depth, max_eval_depth)
            depth_pred = depth_pred.detach()

            mask_1 = (depth_gt > min_eval_depth) * (depth_gt <= 50) * inputs['mask'][:, cam, ...]
            mask_1 = mask_1.bool()
            mask_2 = (depth_gt > 50) * (depth_gt < max_eval_depth) * inputs['mask'][:, cam, ...]
            mask_2 = mask_2.bool()
            
            depth_gt_1 = depth_gt[mask_1]
            depth_pred_1 = depth_pred[mask_1]
            
            # median scale for 0~50m
            scale_val = torch.median(depth_gt_1) / torch.median(depth_pred_1)
            med_scale_1.append(round(scale_val.cpu().numpy().item(), 2))
                            
            depth_pred_metric_1 = torch.clamp(depth_pred_1, min=min_eval_depth, max=max_eval_depth)
            depth_errors_metric_1 = cal_depth_error(depth_pred_metric_1, depth_gt_1)
            
            depth_pred_median_1 = torch.clamp(depth_pred_1 * scale_val, min=min_eval_depth, max=max_eval_depth)
            depth_errors_median_1 = cal_depth_error(depth_pred_median_1, depth_gt_1)

            # (2) 51~200m 범위
            depth_gt_2 = depth_gt[mask_2]
            depth_pred_2 = depth_pred[mask_2]
            
            scale_val = torch.median(depth_gt_2) / torch.median(depth_pred_2)
            med_scale_2.append(round(scale_val.cpu().numpy().item(), 2))
                            
            depth_pred_metric_2 = torch.clamp(depth_pred_2, min=min_eval_depth, max=max_eval_depth)
            depth_errors_metric_2 = cal_depth_error(depth_pred_metric_2, depth_gt_2)
            
            depth_pred_median_2 = torch.clamp(depth_pred_2 * scale_val, min=min_eval_depth, max=max_eval_depth)
            depth_errors_median_2 = cal_depth_error(depth_pred_median_2, depth_gt_2)
            
            
            for i in range(len(depth_errors_metric_1)):
                key = self._metric_names[i]
                error_metric_dict_1[key] += depth_errors_metric_1[i]
                error_median_dict_1[key] += depth_errors_median_1[i]
                error_metric_dict_2[key] += depth_errors_metric_2[i]
                error_median_dict_2[key] += depth_errors_median_2[i]

        if vis_scale==True:
            # print median scale
            print(f'          | median scale (0~50m) = {med_scale_1}        | median scale (50~200m) = {med_scale_2}')
            
                
        for key in error_metric_dict_1.keys():
            error_metric_dict_1[key] = error_metric_dict_1[key].cpu().numpy() / self.num_cams
            error_median_dict_1[key] = error_median_dict_1[key].cpu().numpy() / self.num_cams
            error_metric_dict_2[key] = error_metric_dict_2[key].cpu().numpy() / self.num_cams
            error_median_dict_2[key] = error_median_dict_2[key].cpu().numpy() / self.num_cams
            
        return error_metric_dict_1, error_median_dict_1, error_metric_dict_2, error_median_dict_2         
                
    def print_perf(self, loss, scale): 
        """
        This function prints various metrics for depth estimation accuracy.
        """
        if scale == 'flow':
            perf = f"          | flow_loss = {loss}"
        else:
            perf = ' '*3 + scale
            for k, v in loss.items():
                perf += ' | ' + str(k) + f': {v:.3f}'
        print(perf)

        # 로그 저장
        with open(self.txt_path, "a") as f:
            f.write(perf + "\n")
            