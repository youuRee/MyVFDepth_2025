import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import matplotlib.pyplot as plt

from .misc import cal_depth_error



#@torch.no_grad()  # 필요하면 지우세요 (깊이/포인트로부터 gradient가 필요하면)
def bev_from_depth(
    depth,                # (B,1,H,W)
    invK,                 # (B,4,4) 또는 (B,3,3)도 OK (코드에서 [:3,:3] 사용)
    T_cam2ego=None,       # (B,4,4) extrinsics. 없으면 cam->ego 고정 매핑 사용
    x_bounds=(0.0, 50.0), # 전방(m)
    y_bounds=(-25.0,25.0),# 좌(+)/우(-) 측방(m)
    z_bounds=(-3.0, 3.0), # 높이(Up)(m)
    res=0.2,              # 격자 해상도(m/px)
    agg_channels=("density","z_max","z_mean")  # 원하는 BEV 채널 구성
):
    device = depth.device
    B, _, H, W = depth.shape

    # --- 1) 3D back-projection (카메라 좌표: x=right, y=down, z=forward) ---
    def backproject(invK, depth):
        b, _, h, w = depth.shape
        # pixel grid [2, N]: (u,v)
        xs = torch.arange(w, device=device)
        ys = torch.arange(h, device=device)
        u, v = torch.meshgrid(xs, ys, indexing='xy')
        uv1 = torch.stack([u.reshape(-1), v.reshape(-1), torch.ones(h*w, device=device)], 0)  # [3, N]
        uv1 = uv1.unsqueeze(0).repeat(b,1,1)  # [B,3,N]
        # K^-1 * (uv1) -> [B,3,N]
        Kinv = invK[:, :3, :3]  # [B,3,3]
        rays = Kinv @ uv1
        d = depth.view(b,1,-1)  # [B,1,N]
        pts_cam = rays * d      # [B,3,N]  (x_right, y_down, z_forward)
        # homo
        ones = torch.ones(b,1,pts_cam.shape[-1], device=device)
        return torch.cat([pts_cam, ones], 1)  # [B,4,N]

    homo_cam = backproject(invK, depth)                    # [B,4,N]
    pts_cam = homo_cam[:, :3, :]                           # [B,3,N]

    # --- 2) 카메라->자차(ego) 변환 ---
    # Ego 기준을 x_forward, y_left, z_up 로 두는 게 일반적.
    if T_cam2ego is not None:
        pts_ego = (T_cam2ego @ homo_cam)[:, :3, :]         # [B,3,N]
    else:
        # 고정 매핑(자주 쓰는 관례): x_ego = z_cam, y_ego = -x_cam, z_ego = -y_cam
        x_c, y_c, z_c = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
        pts_ego = torch.stack([z_c, -x_c, -y_c], dim=1)    # [B,3,N]

    x, y, z = pts_ego[:,0], pts_ego[:,1], pts_ego[:,2]     # 각각 [B,N]

    # --- 3) ROI 필터 & 유효 깊이 ---
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    zmin, zmax = z_bounds

    valid = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax) & (z > zmin) & (z < zmax)
    # 깊이값 NaN/Inf/0 제거
    valid = valid & torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)

    # --- 4) 격자 정의 (W_bev: x방향, H_bev: y방향) ---
    W_bev = int(torch.ceil(torch.tensor((xmax - xmin) / res)).item())
    H_bev = int(torch.ceil(torch.tensor((ymax - ymin) / res)).item())

    # --- 5) 인덱싱: (u: x-bin, v: y-bin) ---
    # u in [0, W_bev-1], v in [0, H_bev-1]
    u = ((x - xmin) / res).floor().clamp_(0, W_bev - 1).long()  # [B,N]
    v = ((y - ymin) / res).floor().clamp_(0, H_bev - 1).long()
    lin = (v * W_bev + u)                                       # [B,N]

    # --- 6) 집계 ---
    C = len(agg_channels)
    bev = torch.zeros(B, C, H_bev, W_bev, device=device)

    for b in range(B):
        mask = valid[b]                                         # [N]
        if mask.sum() == 0:
            continue
        lin_b = lin[b, mask]                                    # [M]
        z_b   = z[b, mask]                                      # [M]

        # density (point count)
        if "density" in agg_channels:
            ch = agg_channels.index("density")
            # bincount: lin_b 배열에 있는 각 인덱스(그리드 셀 위치)가 몇 번 등장했는지(몇 개의 점이 속했는지) 계산
            counts = torch.bincount(lin_b, minlength=H_bev*W_bev)  # [H*W]
            bev[b, ch] = counts.float().view(H_bev, W_bev) # 밀도(density): BEV 그리드 셀에 속한 점의 개수
            # 이진 분류 -> 0 or 0이 아닌 것, 다중 분류 or L1 -> (0~1) 사이로 만들어주기
            
            # --- 밀도(density) 정규화 (0~1) ---
            if "density_norm" in agg_channels:
                ch = agg_channels.index("density_norm")
                max_count = bev[b, ch].max()
                if max_count > 0:
                    bev[b, ch] = bev[b, ch] / max_count

        # z_max / z_min (height extremes)
        # PyTorch 2.0+: scatter_reduce
        if "z_max" in agg_channels:
            ch = agg_channels.index("z_max")
            zmax_img = torch.full((H_bev*W_bev,), float("-inf"), device=device)
            # scatter_reduce: 여러 개의 입력값(z_b)을 하나의 출력 위치(lin_b)로 모으면서 특정 연산(amax = 최대값)을 수행
            # -> 같은 그리드 셀에 속한 여러 점들의 z 값(높이) 중에서 최대값을 찾아 해당 셀의 z_max 값으로 설정
            zmax_img = torch.scatter_reduce(zmax_img, 0, lin_b, z_b, reduce='amax', include_self=True)
            bev[b, ch] = zmax_img.view(H_bev, W_bev).clamp_min(zmin)  # 빈셀은 -inf → zmin로 클램프

        if "z_min" in agg_channels:
            ch = agg_channels.index("z_min")
            zmin_img = torch.full((H_bev*W_bev,), float("inf"), device=device)
            zmin_img = torch.scatter_reduce(zmin_img, 0, lin_b, z_b, reduce='amin', include_self=True)
            bev[b, ch] = zmin_img.view(H_bev, W_bev).clamp_max(zmax)

        # z_mean (평균 높이)
        if "z_mean" in agg_channels:
            ch = agg_channels.index("z_mean")
            sum_z = torch.zeros(H_bev*W_bev, device=device)
            # lin_b 인덱스에 해당하는 sum_z 배열의 위치에 z_b 값을 더해줌 -> 각 셀에 속한 점들의 높이(z) 합
            sum_z = sum_z.index_add(0, lin_b, z_b)
            counts = torch.bincount(lin_b, minlength=H_bev*W_bev).clamp_min(1) # 위에서 계산한 점의 개수
            mean_z = (sum_z / counts).view(H_bev, W_bev) # 총합을 개수로 나누어 평균 높이를 계산
            bev[b, ch] = mean_z

    return bev, dict(H=H_bev, W=W_bev, res=res, x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds)


def evaluate_occupancy_iou(gt_depth, pred_depth, invK):
    gt_bev, _ = bev_from_depth(
        gt_depth,                # (B,1,H,W)
        invK,                 # (B,4,4) 또는 (B,3,3)도 OK (코드에서 [:3,:3] 사용)
        T_cam2ego=None,       # (B,4,4) extrinsics. 없으면 cam->ego 고정 매핑 사용
        x_bounds=(0.0, 100.0), # 전방(m)
        y_bounds=(-50.0,50.0),# 좌(+)/우(-) 측방(m)
        z_bounds=(-15.0, 15.0), # 높이(Up)(m)
        res=0.5,              # 격자 해상도(m/px)
        agg_channels=["density"]#("density","z_max","z_mean")  # 원하는 BEV 채널 구성
    )
    
    pred_bev, _ = bev_from_depth(
        pred_depth,           # (B,1,H,W)
        invK,                 # (B,4,4) 또는 (B,3,3)도 OK (코드에서 [:3,:3] 사용)
        T_cam2ego=None,       # (B,4,4) extrinsics. 없으면 cam->ego 고정 매핑 사용
        x_bounds=(0.0, 100.0), # 전방(m)
        y_bounds=(-50.0,50.0),# 좌(+)/우(-) 측방(m)
        z_bounds=(-15.0, 15.0), # 높이(Up)(m)
        res=0.5,              # 격자 해상도(m/px)
        agg_channels=["density"]#("density","z_max","z_mean")  # 원하는 BEV 채널 구성
    )


    thr = 0
    gt_occ   = (gt_bev > thr)
    pred_occ = (pred_bev > thr)

    # 평가 대상 영역: 두 맵의 합집합
    eval_mask = (gt_occ | pred_occ) # gt 또는 pred 중 하나라도 점유된 픽셀들을 포함하는 마스크

    if eval_mask.sum() == 0:
        ious.append(torch.tensor(0., device=gt_bev.device))
        precisions.append(torch.tensor(0., device=gt_bev.device))
        recalls.append(torch.tensor(0., device=gt_bev.device))
        f1s.append(torch.tensor(0., device=gt_bev.device))
        

    g = gt_occ & eval_mask
    p = pred_occ & eval_mask

    tp = (p & g).sum().float()     # 예측과 GT가 모두 점유된 픽셀 (True Positive)
    fp = (p & ~g).sum().float()    # 예측은 점유되었으나 GT는 비어있는 픽셀 (False Positive)
    fn = (~p & g).sum().float()    # 예측은 비어있으나 GT는 점유된 픽셀 (False Negative)
    union = (p | g).sum().float()  # 예측과 GT의 합집합 픽셀 (IoU 계산을 위함)

    iou  = tp / union.clamp(min=1.)
    prec = tp / (tp + fp).clamp(min=1.)
    rec  = tp / (tp + fn).clamp(min=1.)
    f1   = 2*prec*rec / (prec + rec).clamp(min=1e-6)

    return {
            "IoU": iou,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
            }



def evaluate_instance_mean(gt_depth, pred_depth, seg_mask):
    """
    gt_depth:  (1,1,H,W), 0=invalid
    pred_depth:(1,1,H,W)
    seg_mask:  (N,1,H,W), 외부 고정 마스크 권장(SAM/SLIC 등)
    return: metrics(tuple), coverage_ratio
    """
    eps = 1e-6
    # 1) 유효 GT만 사용해 인스턴스 평균
    valid   = (gt_depth > 0).to(gt_depth.dtype)          # (1,1,H,W)
    valid_e = valid.expand_as(seg_mask)                  # (N,1,H,W)
    gt_e    = gt_depth.expand_as(seg_mask)               # (N,1,H,W)

    den = (seg_mask * valid_e).sum(dim=(2,3), keepdim=True)              # (N,1,1,1)
    num = (seg_mask * valid_e * gt_e).sum(dim=(2,3), keepdim=True)       # (N,1,1,1)
    inst_mean = num / (den + eps)                                         # (N,1,1,1)

    # 2) 평균 정의된 인스턴스만 채움
    has_ref = (den > 0).to(inst_mean.dtype)                               # (N,1,1,1)
    inst_filled = seg_mask * inst_mean * has_ref                          # (N,1,H,W)
    inst_filled = inst_filled.sum(dim=0, keepdim=True)                    # (1,1,H,W)

    # 3) 타깃: GT==0인 위치만 인스턴스 평균으로 채움
    invalid = (gt_depth == 0)                                             # (1,1,H,W)
    target  = torch.where(invalid, inst_filled, gt_depth)                 # (1,1,H,W)

    # 4) 평가 마스크: invalid 이면서 채움 성공(inst_filled>0)
    eval_mask = invalid & (inst_filled > 0)
    cov = int(eval_mask.sum())
    tot = int(invalid.sum())
    coverage_ratio = (cov / max(1, tot)) if tot > 0 else 0.0

    if cov == 0:
        # 채워진 invalid 픽셀이 없으면 스킵/0 반환
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = (0,0,0,0,0,0,0)
        coverage_ratio = 0.0
        #return (0,0,0,0,0,0,0), 0.0

    # 5) invalid-only에서 예측/타깃 동시 필터링 후 지표 계산
    pred_m = pred_depth[eval_mask]
    tgt_m  = target[eval_mask]
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = cal_depth_error(pred_m, tgt_m)  # (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    
    return {
            "Abs_rel": abs_rel,
            "Sq_rel": sq_rel,
            "RMSE": rmse,
            "RMSE_log":rmse_log,
            "A1": a1,
            "A2": a2,
            "A3": a3,
            "Cov": coverage_ratio
            }

    