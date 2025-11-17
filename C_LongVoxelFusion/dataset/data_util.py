# Copyright (c) 2023 42dot. All rights reserved.
import os

import numpy as np
import PIL.Image as pil

import torch.nn.functional as F
import torchvision.transforms as transforms

_DEL_KEYS= ['rgb', 'rgb_context', 'rgb_original', 'rgb_context_original', 'intrinsics', 'contexts', 'splitname', 
            'cur_depth', 'prev_depth', 'next_depth', 'prev_pose', 'next_pose', 'point_cloud', 'point_context', 
            'cur_L_pose', 'prev_L_pose', 'next_L_pose'] 


def transform_mask_sample(sample, data_transform):
    """
    This function transforms masks to match input rgb images.
    """
    '''
    image_shape:  (384, 640)
    resize_transform:  Resize(size=(384, 640), interpolation=lanczos, max_size=None, antialias=None)
    '''
    image_shape = data_transform.keywords['image_shape']
    # resize transform
    resize_transform = transforms.Resize(image_shape, interpolation=pil.ANTIALIAS)
    sample['mask'] = resize_transform(sample['mask'])
    # totensor transform
    tensor_transform = transforms.ToTensor()    
    sample['mask'] = tensor_transform(sample['mask'])
    return sample


def img_loader(path):
    """
    This function loads rgb image.
    """
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')


def mask_loader_scene(path, mask_idx, cam):
    """
    This function loads mask that correspondes to the scene and camera.
    """
    fname = os.path.join(path, str(mask_idx), '{}_mask.png'.format(cam.upper()))    
    with open(fname, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('L')


def align_dataset(sample, scales, contexts):
    """
    This function reorganize samples to match our trainer configuration.
    """
    K = sample['intrinsics']
    aug_images = sample['rgb']
    aug_contexts = sample['rgb_context']
    org_images = sample['rgb_original']
    org_contexts= sample['rgb_context_original']    

    n_cam, _, w, h = aug_images.shape

    # initialize intrinsics
    resized_K = np.expand_dims(np.eye(4), 0).repeat(n_cam, axis=0)
    resized_K[:, :3, :3] = K

    # augment images and intrinsics in accordance with scales 
    for scale in scales:
        scaled_K = resized_K.copy()
        scaled_K[:,:2,:] /= (2**scale)
        
        sample[('K', scale)] = scaled_K.copy()
        sample[('inv_K', scale)]= np.linalg.pinv(scaled_K).copy()

        resized_org = F.interpolate(org_images, 
                                          size=(w//(2**scale),h//(2**scale)),
                                          mode = 'bilinear',
                                          align_corners=False)
        resized_aug = F.interpolate(aug_images, 
                                          size=(w//(2**scale),h//(2**scale)), 
                                          mode = 'bilinear',
                                          align_corners=False)            

        sample[('color', 0, scale)] = resized_org
        sample[('color_aug', 0, scale)] = resized_aug
    
    sample[('lidar', 0)] = sample['point_cloud'][0,:,:]    
    sample[('L_pose', 0)] = sample['cur_L_pose']
    sample[('L_pose', -1)] = sample['prev_L_pose']
    sample[('L_pose', 1)] = sample['next_L_pose']
    
    sample[('pose', 0)] = sample['pose']
    sample[('pose', -1)] = sample['prev_pose']
    sample[('pose', 1)] = sample['next_pose']
    
    # 나중에 scale 고려해서 수정
    sample[('gt_depth', 0)] = sample['cur_depth']
    sample[('gt_depth', -1)] = sample['prev_depth']
    sample[('gt_depth', 1)] = sample['next_depth']
    
    # for context data
    for idx, frame in enumerate(contexts):
        sample[('color', frame, 0)] = org_contexts[idx]        
        sample[('color_aug',frame, 0)] = aug_contexts[idx]
        sample[('lidar', frame)] = sample['point_context'][idx][0,:,:]
        
    # delete unused arrays
    for key in list(sample.keys()):
        if key in _DEL_KEYS:
            del sample[key]
    return sample
