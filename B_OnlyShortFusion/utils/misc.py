# Copyright (c) 2023 42dot. All rights reserved.
import os
import yaml 
from collections import defaultdict
import csv
import pandas as pd
import numpy as np
import torch

_NUSC_CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
_DDAD_CAM_LIST = ['camera_01', 'camera_05', 'camera_06', 'camera_07', 'camera_08', 'camera_09']
_REL_CAM_DICT = {0: [1,2], 1: [0,3], 2: [0,4], 3: [1,5], 4: [2,5], 5: [3,4]}


def camera2ind(cameras):
    """
    This function transforms camera name list to indices 
    """    
    indices = []
    for cam in cameras:
        if cam in _DDAD_CAM_LIST:
            ind = _DDAD_CAM_LIST.index(cam)
        elif cam in _NUSC_CAM_LIST:
            ind = _NUSC_CAM_LIST.index(cam)
        else:
            ind = None
        indices.append(ind)
    return indices


def get_relcam(cameras):
    """
    This function returns relative camera indices from given camera list
    """
    relcam_dict = defaultdict(list)
    indices = camera2ind(cameras)
    for ind in indices:
        relcam_dict[ind] = []
        relcam_cand = _REL_CAM_DICT[ind]
        for cand in relcam_cand:
            if cand in indices:
                relcam_dict[ind].append(cand)
    return relcam_dict        
    

def get_config(config, mode='train', weight_path=None):
    """
    This function reads the configuration file and return as dictionary
    """
    with open(config, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

        cfg_name = os.path.splitext(os.path.basename(config))[0]
        print('Experiment: ', cfg_name)

        _log_path = os.path.join(cfg['data']['log_dir'], cfg_name)
        cfg['data']['log_path'] = _log_path
        cfg['data']['save_weights_root'] = os.path.join(_log_path, 'models')
        if weight_path == None:
            weight_path = os.path.join(_log_path, 'models', cfg['load']['weights'])
        cfg['data']['load_weights_dir'] = weight_path
        cfg['data']['num_cams'] = len(cfg['data']['cameras'])
        cfg['model']['mode'] = mode
        cfg['data']['rel_cam_list'] = get_relcam(cfg['data']['cameras'])
        
        if mode == 'train':
            cfg['eval']['syn_visualize'] = False # for pretrained 
            
        elif mode == 'eval':
            cfg['ddp']['world_size'] = 1
            cfg['ddp']['gpus'] = [0]
            cfg['training']['batch_size'] = cfg['eval']['eval_batch_size']
            cfg['training']['depth_flip'] = False
    return cfg

    
def pretty_ts(ts):
    """
    This function prints amount of time taken in user friendly way.
    """
    second = int(ts)
    minute = second // 60
    hour = minute // 60
    return f'{hour:02d}h{(minute%60):02d}m{(second%60):02d}s'

def print_by_depth(pred, target, start, end):
    abs_rel = torch.abs(pred - target) / target
    sq_rel = (pred - target).pow(2) / target

    mask = ((target >= start) & (target < end))

    # abs
    loss = torch.where(mask, abs_rel, 0)
    mean_loss_abs = loss[loss != 0].mean()
    #print("Loss")
    #print("Abs Rel: {:.4f}".format(mean_loss_abs), end=" ")

    # sq
    loss = torch.where(mask, sq_rel, 0)
    mean_loss_sq = loss[loss != 0].mean()
    #print("Sq Rel: {:.4f}".format(mean_loss_sq), end=" ")

    new_target = torch.where(mask, target, 0)
    new_pred = torch.where(mask, pred, 0)
    new_target = new_target[new_target != 0]
    new_pred = new_pred[new_pred != 0]

    rmse = torch.sqrt(torch.mean((new_pred-new_target).pow(2)))
    rmse_log = torch.sqrt(torch.mean((torch.log(new_target) - torch.log(new_pred)).pow(2)))
    #print("RMSE: {:.4f}".format(rmse), end=" ")
    #print("RMSElog: {:.4f}".format(rmse_log))
    
    thresh = torch.max((new_target / new_pred), (new_pred / new_target))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()
    #print("Accuracy")
    #print("a1: {:.4f}, a2: {:.4f}, a3: {:.4f}\n".format(a1.item(), a2.item(), a3.item()))

    # csv 파일에 저장
    output_file = 'output_depth.csv'
    
    new_data = {
        "Depth" : [end],
        "Abs.Rel": [mean_loss_abs.item()],
        "Sq.Rel": [mean_loss_sq.item()],
        "RMSE": [rmse.item()],
        "RMSElog": [rmse_log.item()],
        "a1": [a1.item()],
        "a2": [a2.item()],
        "a3": [a3.item()]
    }

    # 데이터프레임 생성
    df = pd.DataFrame(new_data)

    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(output_file, index=False, mode='a', encoding='utf-8-sig', header=False)


def cal_depth_error(pred, target):
    """
    This function calculates depth error using various metrics.
    """
    # target : torch.Size([26130]), cuda:0
    
    abs_rel = torch.mean(torch.abs(pred-target) / target)
    sq_rel = torch.mean((pred-target).pow(2) / target)
    rmse = torch.sqrt(torch.mean((pred-target).pow(2)))
    rmse_log = torch.sqrt(torch.mean((torch.log(target) - torch.log(pred)).pow(2)))

    thresh = torch.max((target/pred), (pred/ target))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()
    '''
    depth_list = np.arange(0, 201, 5)
    for i in range(1, len(depth_list)):
        print_by_depth(pred, target, depth_list[i-1], depth_list[i])
    '''
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3