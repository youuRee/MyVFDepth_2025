# Copyright (c) 2023 42dot. All rights reserved.
import time
import math
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist

from utils import Logger
import os

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
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        model.set_train()
        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time()
            model.optimizer.zero_grad(set_to_none=True)
            outputs, losses = model.process_batch(inputs, self.rank)
            losses['total_loss'].backward()
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
        
        if 'depth' in inputs:
            syn_depth = self.syn_visualize
            depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, syn_depth, vis_scale=True)
            self.logger.print_perf(depth_eval_metric, 'metric')
            self.logger.print_perf(depth_eval_median, 'median')

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
        
        process = tqdm(eval_dataloader)
        
        #highway_scene_list = ['000160', '000161', '000161', '000163', '000164', '000165', '000166', '000167', '000168', '000169', '000170', '000172', '000174', '000175', '000176']
        #vulnerable_scene_list = ['000151', '000164', '000169', '000172', '000175', '000176', '000180', '000183', '000188', '000189', '000195', '000196']
        
        for batch_idx, inputs in enumerate(process):   
            # visualize synthesized depth maps
            #if self.syn_visualize and batch_idx < self.syn_idx:
            #    continue
            
            forder_name = inputs['filename'][0].split('/')[0]
            #if forder_name in highway_scene_list:
            #    forder_name += ' (highway)'
            
            #if forder_name not in vulnerable_scene_list:
            #    continue

            #print("Scene: ", forder_name, "!!!") 
            outputs, _ = model.process_batch(inputs, self.rank)
            
            # 전체 딕셔너리 저장
            #out_path = '/workspace/MyVFDepth-LongRange2/outputs/' + forder_name + '-' + str(batch_idx) + ".pt"
            #out_path = os.path.join('/workspace/MyVFDepth-LongRange/outputs/', '{}'.format(batch_idx) + ".pt")
            #torch.save(outputs, out_path)
            #in_path = '/workspace/MyVFDepth-LongRange2/inputs/' + forder_name + '-' + str(batch_idx) + ".pt"
            #in_path = os.path.join('/workspace/MyVFDepth-LongRange/inputs/', '{}'.format(batch_idx) + ".pt")
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
                
                
                perf = 'scene = ' + forder_name
                perf += '\nmetric | ' + ' | '.join([f'{k}: {v:.3f}' for k, v in depth_eval_metric.items()])
                perf += '\nmedian | ' + ' | '.join([f'{k}: {v:.3f}' for k, v in depth_eval_median.items()])
                
                with open('/workspace/MyVFDepth-SimpleFusion/scene_by_results_gtpose_synth_30m.txt', "a") as f:
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
            for key in self.depth_metric_names:
                avg_depth_eval_metric[key] /= len(eval_dataloader)
                avg_depth_eval_median[key] /= len(eval_dataloader)

            print('Evaluation result...\n')
            self.logger.print_perf(avg_depth_eval_metric, 'metric')
            self.logger.print_perf(avg_depth_eval_median, 'median')
