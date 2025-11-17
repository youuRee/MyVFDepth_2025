import torch.nn as nn
import torch.nn.functional as F

class RayEncoding(nn.Module):
    def __init__(self, in_channels, out_channels, depth_bins):
        super(RayEncoding, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.depth_bins = depth_bins

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.conv3d(x)  # 기하학적 특징 반영
        B, C, D, H, W = x.shape
        x = x.view(B, C * D, H, W)  # 2D 디코더 호환을 위한 flatten
        return x

