# Copyright (c) 2023 42dot. All rights reserved.
import time
import math
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist

from utils import Logger
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm

import numpy as np
import torchvision.utils as vutils
import cv2

def score_map_vis(score_map, cmap='bone', vminmax=None, max_perc=95):
    """ Accepts score_map as torch.Tensor of shape [1, 1, h, w] or np.ndarray of shape [h, w]
        Assumes the image images uses bgr format (loaded from cv2 instead of PIL)
    """
    score_map_np = score_map.squeeze().detach().cpu().numpy() if torch.is_tensor(score_map) else score_map   # either torch.Tensor or np.ndarray

    if vminmax == None:
        vmin = score_map_np.min()
        vmax = np.percentile(score_map_np, max_perc)
    else:
        vmin, vmax = vminmax

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    return mapper.to_rgba(score_map_np)[:, :, :3]

def hsv_to_rgb(image):
    """ Convert image from hsv to rgb color space, input must be torch.Tensor of shape (*, 3, H, W)
    """
    assert isinstance(image, torch.Tensor), f"Input type is not a torch.Tensor. Got {type(image)}"
    assert len(image.shape) >= 3 and image.shape[-3] == 3, f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"

    h = image[..., 0, :, :]
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]
    

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()  # turns very negative for nan
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out

def cart2polar(cart):
    """ Convert cartian points into polar coordinates, the last dimension of the input must contain y and x component
    """
    assert cart.shape[-1] == 2, 'Last dimension must contain y and x vector component'

    r = torch.sqrt(torch.sum(cart**2, -1))
    theta = torch.atan(cart[...,0]/cart[...,1])
    theta[torch.where(torch.isnan(theta))] = 0  # torch.atan(0/0) gives nan

    theta[cart[...,1] < 0] += torch.pi
    theta = (5*torch.pi/2 - theta) % (2*torch.pi)

    return r, theta

def vis_motion(pix_motion_raw):
    mag, theta = cart2polar(pix_motion_raw)
    max_mag = (mag.max().item() + 1e-8)
    hsv = torch.ones(1, 3, 384, 640).to('cuda')
    hsv[:, 0] = (theta - torch.pi/4) % (2 * torch.pi) / (2*torch.pi)
    hsv[:, 1] = 1.0
    hsv[:, 2] = mag / max_mag
    motion_visual = 1 - hsv_to_rgb(hsv)
    
    return motion_visual[0].permute(1, 2, 0).detach().cpu().numpy()


def visualize_motion_images(depth, motion_mask, color,
                            sample_ego, sample_complete, sample,
                            title_prefix='', save_path=None, figsize=(15,10)):
     # 2행 3열의 subplot 생성
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    
    axs[0, 0].imshow(color)
    axs[0, 0].set_title(f"{title_prefix}RGB")
    axs[0, 0].axis('off')
    
    # Subplot 1: Depth (보통 그레이스케일)
    axs[0, 1].imshow(depth)
    axs[0, 1].set_title(f"{title_prefix}Depth")
    axs[0, 1].axis('off')
    
    # Subplot 2: Motion Mask (그레이스케일)
    axs[0, 2].imshow(motion_mask)
    axs[0, 2].set_title(f"{title_prefix}Motion Mask")
    axs[0, 2].axis('off')
    
    # Subplot 3: Independent Flow (컬러)
    #axs[0, 2].imshow(ind_flow)
    #axs[0, 2].set_title(f"{title_prefix}Independent Flow")
    #axs[0, 2].axis('off')
    
    # Subplot 4: Sample Ego
    axs[1, 0].imshow(sample_ego)
    axs[1, 0].set_title(f"{title_prefix}Sample Ego")
    axs[1, 0].axis('off')
    
    # Subplot 5: Sample Complete
    axs[1, 1].imshow(sample_complete)
    axs[1, 1].set_title(f"{title_prefix}Sample Complete")
    axs[1, 1].axis('off')
    
    # Subplot 6: Sample
    axs[1, 2].imshow(sample)
    axs[1, 2].set_title(f"{title_prefix}Sample")
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()

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


def plot_disparity_vs_depth(pred_disp, gt_depth, save_path, max_depth=30, window=500):
    """
    pred_disp : [B, 1, H, W] 예측 disparity (torch.Tensor)
    gt_depth  : [B, 1, H, W] GT depth (torch.Tensor)
    max_depth : x축 마지막 범위 (m)
    window    : 이동평균 윈도우 크기
    """

    # CPU numpy 변환
    disp = pred_disp.detach().cpu().numpy().reshape(-1)
    depth = gt_depth.detach().cpu().numpy().reshape(-1)

    # 유효 depth 마스크 (0 < depth <= max_depth)
    mask = (depth > 0) & (depth <= max_depth)
    disp = disp[mask]
    depth = depth[mask]

    # 깊이에 따라 정렬
    sort_idx = np.argsort(depth)
    depth = depth[sort_idx]
    disp = disp[sort_idx]

    # disparity 변화율 (근사: numpy gradient 사용)
    dd_dz = np.gradient(disp, depth)

    # Figure
    plt.figure(figsize=(10, 5))

    # Depth–Disparity 산점도 & 평균 곡선
    plt.subplot(1, 2, 1)
    plt.scatter(depth, disp, s=1, alpha=0.2, label="Pixel samples")

    if len(disp) > window:
        avg_disp = np.convolve(disp, np.ones(window)/window, mode='valid')
        avg_depth = depth[window-1:]  # 길이를 avg_disp와 맞춤
        plt.plot(avg_depth, avg_disp, color="red", label="Smoothed curve")

    plt.xlabel("Depth (m)")
    plt.ylabel("Predicted Disparity (px)")
    plt.title("Depth vs Disparity")
    plt.legend()
    plt.grid(True)

    # Disparity 변화율
    plt.subplot(1, 2, 2)
    plt.plot(depth, dd_dz, color="blue", alpha=0.7, label="d(Disparity)/dZ")
    plt.xlabel("Depth (m)")
    plt.ylabel("Disparity Gradient (px/m)")
    plt.title("Disparity Gradient vs Depth")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class VFDepthTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def learn(self, model):
        """
        This function sets training process.
        """        
        train_dataloader = model.train_dataloader()
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()


        for self.epoch in range(self.num_epochs):

            if self.ddp_enable:
                model.train_sampler.set_epoch(self.epoch) 
                
            self.train(model, train_dataloader, start_time)
            
            # save model after each epoch using rank 0 gpu 
            if self.rank == 0:
                model.save_model(self.epoch)
                print('-'*110) 
                
            if self.ddp_enable:
                dist.barrier()
            
            #torch.cuda.empty_cache()
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        
        #model.bool_CmpFlow = self.epoch > 0
        model.bool_MotMask = False#self.epoch > 0
        model.bool_Depth = True
        model.bool_CmpFlow = False
        
        '''
        if self.epoch == 0:
            model.start_flag = True
        else:
            model.start_flag = False
        
        # 🔹 5-epoch 주기: 첫 번째는 depth 학습, 나머지 4는 flow 학습
        cycle = self.epoch % 10  # 0~4 순환

        if cycle > 5:
            # 주기 첫 epoch: depth만 학습
            model.bool_Depth = True
            model.bool_CmpFlow = False
            print(f"[Epoch {self.epoch}] Training depth network only.")
            for param in model.models['depth_net'].parameters():
                param.requires_grad = True
            for param in model.models['refine_net'].parameters():
                param.requires_grad = False
        else:
            # 나머지 4 epoch: flow만 학습
            model.bool_Depth = False
            model.bool_CmpFlow = True
            print(f"[Epoch {self.epoch}] Freezing depth network, training flow network only.")
            for param in model.models['depth_net'].parameters():
                param.requires_grad = False
            for param in model.models['refine_net'].parameters():
                param.requires_grad = True
        '''
        model.set_train()
        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time()
            model.optimizer.zero_grad(set_to_none=True)
            outputs, losses = model.process_batch(inputs, self.rank)
            
            loss = losses['total_loss']
            if torch.isnan(loss):
                print(f"[WARNING] NaN loss at step {self.step}, skipping.")
                torch.save(inputs, f"/workspace/MyVFDepth-LongRange2/NaN_Loss/step_{self.step}.pt")
                #model.zero_grad(set_to_none=True)
                del loss, outputs
                torch.cuda.empty_cache()
                continue
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"[BACKWARD ERROR] NaN in backward at step {self.step}: {e}")
                continue

            #losses['total_loss'].backward()
            
            '''
            # DDP 모드에 따라서 모델 파라미터에 접근
            trained_parameters = []
            # VFDepthAlgo의 models 객체가 torch.nn.Module 클래스를 상속 받음 -> 파라미터 정보를 가짐 (named_parameters 사용 가능)
            for net_name, net in model.models.items():
                if model.ddp_enable and hasattr(net, 'module'):
                    net_params = net.module.named_parameters()
                else:
                    net_params = net.named_parameters()

                trained_parameters += [name for name, param in net_params if param.grad is not None]

            print(f"Trained parameters at step {self.step}: {trained_parameters}")
            '''
            model.optimizer.step()

            if self.rank == 0: 
                self.logger.update(
                    'train', 
                    self.epoch, 
                    self.world_size,
                    batch_idx, 
                    self.step,
                    start_time,
                    before_op_time, 
                    inputs,
                    outputs,
                    losses
                )

                if self.logger.is_checkpoint(self.step):
                    self.validate(model)

            if self.ddp_enable:
                dist.barrier()

            self.step += 1

        model.lr_scheduler.step()

        
    @torch.no_grad()
    def validate(self, model):
        """
        This function validates models on validation dataset to monitor training process.
        """
        model.set_val()
        inputs = next(self.val_iter)
            
        outputs, losses = model.process_batch(inputs, self.rank)
        
        scale = 0
        for cam in range(6):
            
            for frame_id in self.frame_ids[1:]:
                if frame_id < 0:
                    name = 'prev'
                else:
                    name = 'next'
            
                path = '/workspace/MyVFDepth_2025/D_LongBinFusion_Large/logs/cam' + str(cam) + '/' + str(self.step) + 'step_' + name + '.png'
                
                gt_img = inputs[('color', 0, scale)][:, cam, ...][0].permute(1, 2, 0).detach().cpu().numpy()
                warp_img = outputs[('cam', cam)][('color', frame_id, scale)][0].permute(1, 2, 0).detach().cpu().numpy()
                gt_depth = inputs[('gt_depth', 0)][:, cam, ...][0].permute(1, 2, 0).detach().cpu().numpy()
                depth = outputs[('cam', cam)][('depth', 0)][0].permute(1, 2, 0).detach().cpu().numpy()
                
                fig, axs = plt.subplots(2, 2, figsize=(20, 15))
                fig.suptitle(f"Frame {frame_id} - Camera {cam}", fontsize=14)

                # 1. Ground truth 이미지
                axs[0, 0].imshow(gt_img)
                axs[0, 0].set_title("GT Image")
                axs[0, 0].axis('off')

                # 2. Warp된 이미지
                axs[0, 1].imshow(warp_img)
                axs[0, 1].set_title("Warped Image (t'->t)")
                axs[0, 1].axis('off')

                # 4. Depth
                axs[1, 0].imshow(gt_depth)
                axs[1, 0].set_title("GT Depth")
                axs[1, 0].axis('off')

                # 5. Warp된 Depth
                axs[1, 1].imshow(depth)
                axs[1, 1].set_title("Pred Depth")
                axs[1, 1].axis('off')

                plt.tight_layout()
                plt.savefig(path)
                plt.close()
            
        
        if 'depth' in inputs:
            #depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, vis_scale=True)
            syn_depth = self.syn_visualize
            depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, syn_depth, vis_scale=True)
            self.logger.print_perf(depth_eval_metric, 'metric')
            self.logger.print_perf(depth_eval_median, 'median')
            #self.logger.print_perf(losses['flow_loss'], 'flow')

        self.logger.log_tb('val', inputs, outputs, losses, self.step)            
        del inputs, outputs, losses
        
        model.set_train()
    
    @torch.no_grad()
    def evaluate(self, model, vis_results=False):
        """
        This function evaluates models on full validation dataset.
        """
        eval_dataloader = model.eval_dataloader()
        
        # load model
        #print(model.weight_path)
        model.load_weights()
        model.set_val()
        model.bool_CmpFlow = True
        model.bool_MotMask = True
        
        avg_depth_eval_metric = defaultdict(float)
        avg_depth_eval_median = defaultdict(float)  
        
        avg_depth_eval_metric_1 = defaultdict(float)
        avg_depth_eval_median_1 = defaultdict(float)        
        avg_depth_eval_metric_2 = defaultdict(float)
        avg_depth_eval_median_2 = defaultdict(float)       
        
        count_metric_2 = defaultdict(int)  # ← 각 metric 별로 카운트 
        count = 0
        
        process = tqdm(eval_dataloader)
        
        #highway_scene_list = ['000160', '000161', '000161', '000163', '000164', '000165', '000166', '000167', '000168', '000169', '000170', '000172', '000174', '000175', '000176']
        #vulnerable_scene_list = ['000151', '000164', '000169', '000172', '000175', '000176', '000180', '000183', '000188', '000189', '000195', '000196']
        #scene_list = ['000151', '000157', '000159', '000160', '000171', '000174', '000189', '000191', '000193', '000195', '000196', '000199']
        for batch_idx, inputs in enumerate(process):   
            # visualize synthesized depth maps
            #if self.syn_visualize and batch_idx < self.syn_idx:
            #    continue
            
            outputs, _ = model.process_batch(inputs, self.rank)
            
            # 전체 딕셔너리 저장
            #out_path = '/workspace/MyVFDepth-LongRange2/outputs/' + forder_name + '-' + str(batch_idx) + ".pt"
            #torch.save(outputs, out_path)
            #in_path = '/workspace/MyVFDepth-LongRange2/inputs/' + forder_name + '-' + str(batch_idx) + ".pt"
            #torch.save(inputs, in_path)
                        
            if self.eval_interval:
                depth_eval_metric_1, depth_eval_median_1, depth_eval_metric_2, depth_eval_median_2 = self.logger.compute_depth_losses_eval(inputs, outputs)
            
                for key in self.depth_metric_names:
                    avg_depth_eval_metric_1[key] += depth_eval_metric_1[key]
                    avg_depth_eval_median_1[key] += depth_eval_median_1[key]
                    
                    if not math.isnan(depth_eval_metric_2[key]):
                        avg_depth_eval_metric_2[key] += depth_eval_metric_2[key]
                        avg_depth_eval_median_2[key] += depth_eval_median_2[key]
                        count_metric_2[key] += 1  # ← 카운트 누적
                        #print(f"{key} count: {count_metric_2[key]}")
                
                
                if vis_results:
                    self.logger.log_result(inputs, outputs, batch_idx, self.syn_visualize)
                
                if self.syn_visualize and batch_idx >= self.syn_idx:
                    process.close()
                    break

            else:
                syn_depth = self.syn_visualize
                depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, syn_depth)
                
                for key in self.depth_metric_names:
                    avg_depth_eval_metric[key] += depth_eval_metric[key]
                    avg_depth_eval_median[key] += depth_eval_median[key]
                
                '''
                scale = 0
                forder_name = inputs['filename'][0].split('/')[0]
                for cam in range(6):
                    path = '/workspace/MyVFDepth-LongRange2/disparity_diff/cam' + str(cam) + '/' + forder_name + '-' + str(batch_idx) + '.png'
                    gt_depth = inputs[('gt_depth', 0)][:, cam, ...]#[0].detach().cpu()
                    pred_disp = outputs[('cam', cam)][('disp', 0)]
                    plot_disparity_vs_depth(pred_disp, gt_depth, path, max_depth=10, window=500)
                
                perf = 'scene = ' + forder_name
                perf += '\nmetric | ' + ' | '.join([f'{k}: {v:.3f}' for k, v in depth_eval_metric.items()])
                perf += '\nmedian | ' + ' | '.join([f'{k}: {v:.3f}' for k, v in depth_eval_median.items()])
                
                with open('/workspace/MyVFDepth-LongRange2/scene_by_results_objloss_10m.txt', "a") as f:
                    f.write(perf + "\n")
                '''
                if vis_results:
                    self.logger.log_result(inputs, outputs, batch_idx, self.syn_visualize)
                
                if self.syn_visualize and batch_idx >= self.syn_idx:
                    process.close()
                    break
            
            
        if self.eval_interval:
            for key in self.depth_metric_names:
                avg_depth_eval_metric_1[key] /= len(eval_dataloader)
                avg_depth_eval_median_1[key] /= len(eval_dataloader)
                avg_depth_eval_metric_2[key] /= count_metric_2[key]
                avg_depth_eval_median_2[key] /= count_metric_2[key]

            print('Evaluation result... (0~50m)\n')
            self.logger.print_perf(avg_depth_eval_metric_1, 'metric')
            self.logger.print_perf(avg_depth_eval_median_1, 'median')
            
            print('Evaluation result... (50~200m), Scene: ', count_metric_2, '\n')
            self.logger.print_perf(avg_depth_eval_metric_2, 'metric')
            self.logger.print_perf(avg_depth_eval_median_2, 'median')
        
        else:
            for key in self.depth_metric_names:
                avg_depth_eval_metric[key] /= len(eval_dataloader)
                avg_depth_eval_median[key] /= len(eval_dataloader)

            print('Evaluation result...\n')
            self.logger.print_perf(avg_depth_eval_metric, 'metric')
            self.logger.print_perf(avg_depth_eval_median, 'median')
    
    '''   
    @torch.no_grad()
    def evaluate(self, model, vis_results=False):
        """
        This function evaluates models on full validation dataset.
        """
        eval_dataloader = model.eval_dataloader()
        
        # load model
        #print(model.weight_path)
        model.load_weights()
        model.set_val()
        model.bool_CmpFlow = True
        model.bool_MotMask = True
        
        avg_inst_depth_eval = defaultdict(float)
        avg_bev_iou_eval = defaultdict(float)  
            
        
        count_metric = defaultdict(int)  # ← 각 metric 별로 카운트 
        
        process = tqdm(eval_dataloader)
        
        #highway_scene_list = ['000160', '000161', '000161', '000163', '000164', '000165', '000166', '000167', '000168', '000169', '000170', '000172', '000174', '000175', '000176']
        #vulnerable_scene_list = ['000151', '000164', '000169', '000172', '000175', '000176', '000180', '000183', '000188', '000189', '000195', '000196']
        
        for batch_idx, inputs in enumerate(process):   
            
            outputs, _ = model.process_batch(inputs, self.rank)
                        
            if self.eval_interval:
                depth_eval_metric_1, depth_eval_median_1, depth_eval_metric_2, depth_eval_median_2 = self.logger.compute_depth_losses_eval(inputs, outputs)
            
                for key in self.depth_metric_names:
                    avg_depth_eval_metric_1[key] += depth_eval_metric_1[key]
                    avg_depth_eval_median_1[key] += depth_eval_median_1[key]
                    
                    if not math.isnan(depth_eval_metric_2[key]):
                        avg_depth_eval_metric_2[key] += depth_eval_metric_2[key]
                        avg_depth_eval_median_2[key] += depth_eval_median_2[key]
                        count_metric_2[key] += 1  # ← 카운트 누적
                        #print(f"{key} count: {count_metric_2[key]}")
                
                
                if vis_results:
                    self.logger.log_result(inputs, outputs, batch_idx, self.syn_visualize)
                
                if self.syn_visualize and batch_idx >= self.syn_idx:
                    process.close()
                    break

            else:
                syn_depth = self.syn_visualize
                error_inst_dict, iou_bev_dict = self.logger.compute_invalid_depth_losses(inputs, outputs, syn_depth)
                
                for key in error_inst_dict:
                    if not math.isnan(error_inst_dict[key]):
                        avg_inst_depth_eval[key] += error_inst_dict[key]
                        count_metric[key] += 1  # ← 카운트 누적
                    
                for key in iou_bev_dict:    
                    avg_bev_iou_eval[key] += iou_bev_dict[key]
                
                forder_name = inputs['filename'][0].split('/')[0]
                perf = 'scene = ' + forder_name
                perf += '\ndepth | ' + ' | '.join([f'{k}: {v:.3f}' for k, v in error_inst_dict.items()])
                perf += '\nbev | ' + ' | '.join([f'{k}: {v:.3f}' for k, v in iou_bev_dict.items()])
                with open('/workspace/MyVFDepth-LongRange2/scene_by_results_invalid_objwise.txt', "a") as f:
                    f.write(perf + "\n")
                
                if vis_results:
                    self.logger.log_result(inputs, outputs, batch_idx, self.syn_visualize)
                
                if self.syn_visualize and batch_idx >= self.syn_idx:
                    process.close()
                    break
            
            
        if self.eval_interval:
            for key in self.depth_metric_names:
                avg_depth_eval_metric_1[key] /= len(eval_dataloader)
                avg_depth_eval_median_1[key] /= len(eval_dataloader)
                avg_depth_eval_metric_2[key] /= count_metric_2[key]
                avg_depth_eval_median_2[key] /= count_metric_2[key]

            print('Evaluation result... (0~50m)\n')
            self.logger.print_perf(avg_depth_eval_metric_1, 'metric')
            self.logger.print_perf(avg_depth_eval_median_1, 'median')
            
            print('Evaluation result... (50~200m), Scene: ', count_metric_2, '\n')
            self.logger.print_perf(avg_depth_eval_metric_2, 'metric')
            self.logger.print_perf(avg_depth_eval_median_2, 'median')
        
        else:
            for key in avg_inst_depth_eval:
                avg_inst_depth_eval[key] /= count_metric[key]#len(eval_dataloader)
            for key in avg_bev_iou_eval:
                avg_bev_iou_eval[key] /= len(eval_dataloader)
            
            print('Evaluation result...\n')
            #self.logger.print_perf(avg_inst_depth_eval, 'metric')
            
            print('Difference Instance Mean Depth (Scene: ', count_metric, ')')
            for key in avg_inst_depth_eval:
                print(key,': ', avg_inst_depth_eval[key], end=" | ")
            
            print('\n\nIoU in BEV Occupancy')
            for key in avg_bev_iou_eval:
                print(key,': ', avg_bev_iou_eval[key], end=" | ")
            print()
    
    '''