# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn.functional as F
    

def compute_auto_masks(reprojection_loss, identity_reprojection_loss):
    """ 
    This function computes auto mask using reprojection loss and identity reprojection loss.
    """
    if identity_reprojection_loss is None:
        # without using auto(identity loss) mask
        reprojection_loss_mask = torch.ones_like(reprojection_loss)
    else:
        # using auto(identity loss) mask
        # reprojection_loss < identity_reprojection_loss → 와핑이 정확 → 유효 (1)
        # identity_reprojection_loss < reprojection_loss → 정적일 가능성이 높음 (ego car 안움직였다거나...) → 무효 (0)
        losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
        idxs = torch.argmin(losses, dim=1, keepdim=True)
        reprojection_loss_mask = (idxs == 0).float()
    return reprojection_loss_mask


def compute_masked_loss(loss, mask):    
    """
    This function masks losses while avoiding zero division.
    """    
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def compute_edg_smooth_loss(rgb, disp_map):
    """
    This function calculates edge-aware smoothness.
    """
    '''
    RGB 이미지를 사용해서 Disparity Map 스무딩 하는 이유
    
    1. 에지 정보 보존 (Edge Preservation): 
     - RGB 이미지의 픽셀 값 변화(즉, 기울기)는 물체의 경계(에지)를 잘 나타냄
     -> 에지 근처에서는 Disparity Map이 급격히 변화할 수 있어야 하므로, 에지 영역에서는 스무딩을 억제
     -> 에지가 아닌 부드러운 영역에서는 Disparity Map을 더 매끄럽게
       
    2. 에지 민감도 반영 (Edge-Aware Regularization):
     - RGB 기울기(grad_rgb_x, grad_rgb_y)가 큰 경우(에지 영역) -> 스무딩 정도를 낮추기 위해 exp(-grad_rgb) 값을 통해 스무딩 손실에 작은 가중치 부여
     - RGB 기울기가 작은 경우(평탄한 영역) -> 스무딩 손실에 더 큰 가중치를 부여하여 Disparity Map을 매끄럽게 조정
    '''
    
    # RGB 이미지의 수평(grad_rgb_x)과 수직(grad_rgb_y) 기울기를 계산 ->  RGB 이미지에서 물체 경계(에지) 정보를 나타냄
    grad_rgb_x = (rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).abs().mean(1, True) # shape = (b,c,h,w)
    grad_rgb_y = (rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).abs().mean(1, True)

    # Disparity Map의 수평(grad_disp_x) 및 수직(grad_disp_y) 기울기를 계산 -> Disparity Map의 변화 정도를 나타냄
    grad_disp_x = (disp_map[:, :, :, :-1] - disp_map[:, :, :, 1:]).abs()
    grad_disp_y = (disp_map[:, :, :-1, :] - disp_map[:, :, 1:, :]).abs()

    # RGB 이미지의 기울기(grad_rgb_x, grad_rgb_y)를 기반으로 Disparity Map의 기울기 값을 조정 -> RGB 기울기가 작을수록 이 값은 커짐(스무딩 강화)
    grad_disp_x *= (-1.0 * grad_rgb_x).exp()
    grad_disp_y *= (-1.0 * grad_rgb_y).exp()
    return grad_disp_x.mean() + grad_disp_y.mean()


def compute_ssim_loss(pred, target):
    """
    This function calculates SSIM loss between predicted image and target image.
    """
    ref_pad = torch.nn.ReflectionPad2d(1)
    pred = ref_pad(pred)
    target = ref_pad(target)

    mu_pred = F.avg_pool2d(pred, kernel_size = 3, stride = 1)
    mu_target = F.avg_pool2d(target, kernel_size = 3, stride = 1)

    musq_pred = mu_pred.pow(2)
    musq_target = mu_target.pow(2)
    mu_pred_target = mu_pred*mu_target

    sigma_pred = F.avg_pool2d(pred.pow(2), kernel_size = 3, stride = 1)-musq_pred
    sigma_target = F.avg_pool2d(target.pow(2), kernel_size = 3, stride = 1)-musq_target
    sigma_pred_target = F.avg_pool2d(pred*target, kernel_size = 3, stride = 1)-mu_pred_target

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu_pred_target + C1)*(2*sigma_pred_target + C2)) \
                    /((musq_pred + musq_target + C1)*(sigma_pred + sigma_target + C2)+1e-8)    
    return torch.clamp((1-ssim_map)/2, 0, 1)


def compute_photometric_loss(pred=None, target=None):
    """
    This function calculates photometric reconstruction loss (0.85*SSIM + 0.15*L1)
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    ssim_loss = compute_ssim_loss(pred, target).mean(1, True)
    rep_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    return rep_loss
