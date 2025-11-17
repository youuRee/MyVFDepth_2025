import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def get_canny_edge(rgb_tensor):
    """
    rgb_tensor: [B, 3, H, W] torch.Tensor in range [0, 1]
    return: edge map [B, 1, H, W] torch.Tensor in float
    """
    edge_maps = []
    for i in range(rgb_tensor.size(0)):
        img_np = rgb_tensor[i].cpu().numpy().transpose(1, 2, 0) * 255  # [H,W,3], range [0,255]
        img_np = img_np.astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 100, 200)  # you can tune thresholds
        edge_tensor = torch.from_numpy(edge / 255.0).float().unsqueeze(0)  # [1,H,W]
        edge_maps.append(edge_tensor)
    
    edge_maps = torch.stack(edge_maps, dim=0).to(rgb_tensor.device)  # [B,1,H,W]
    return edge_maps

def compute_motion_gradient(motion_prob):
    """
    motion_prob: [B,1,H,W]
    return: gradient magnitude [B,1,H,W]
    """
    grad_x = torch.abs(motion_prob[:, :, :, :-1] - motion_prob[:, :, :, 1:])
    grad_x = F.pad(grad_x, (0,1,0,0), mode='replicate')  # 오른쪽 padding: width +1
    
    grad_y = torch.abs(motion_prob[:, :, :-1, :] - motion_prob[:, :, 1:, :])
    grad_y = F.pad(grad_y, (0,0,0,1), mode='replicate')  # 아래쪽 padding: height +1
    
    grad = grad_x + grad_y
    return grad

def edge_guided_motion_loss(motion_prob, rgb, lambda_edge=1.0, lambda_smooth=0.5):
    edge = get_canny_edge(rgb)  # [B,1,H,W], float in [0,1]
    grad_motion = compute_motion_gradient(motion_prob)  # [B,1,H,W]

    # Edge 위치에서 gradient 커야 함
    edge_loss = F.l1_loss(grad_motion * edge, edge)

    # Edge 아닌 곳에서는 gradient 작아야 함 (스무딩)
    smooth_loss = (grad_motion * (1 - edge)).abs().mean()
    
    loss = lambda_edge * edge_loss + lambda_smooth * smooth_loss
    
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Loss : {loss:.3f}", fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.imshow(edge[0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
    plt.title("Canny Edge")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    im = plt.imshow(grad_motion[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.title("Motion Map Grad")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'motion_gradient.png')
    plt.close()

    return loss
