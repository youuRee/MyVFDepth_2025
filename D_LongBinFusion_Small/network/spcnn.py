import numpy as np
import torch
import spconv.pytorch as spconv
from torch import nn
from spconv.pytorch import functional as Fsp
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.hash import HashTable


# second or voxelnet local feat
class SpconvNet(spconv.SparseModule):
    def __init__(self, sp_dim):
        super(SpconvNet, self).__init__()
        
        self.block1 = spconv.SparseSequential(
            spconv.SparseConv3d(64, 32, 3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            spconv.SparseConv3d(32, sp_dim, 3, padding=1),
            torch.nn.BatchNorm1d(sp_dim),
            torch.nn.ReLU()
        )
        self.block2 = spconv.SparseSequential(
            spconv.SparseConv3d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            spconv.SparseConv3d(in_channels=32, out_channels=sp_dim, kernel_size=2, stride=2),
            torch.nn.BatchNorm1d(sp_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, x, types):
        if types == 'short':
            x = self.block2(x)
        else:
            x = self.block1(x)
        return x


'''
# second or voxelnet local feat
class SpconvNet(spconv.SparseModule):
    def __init__(self, sp_dim):
        super(SpconvNet, self).__init__()
        
        self.block_long = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, 3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            spconv.SparseConv3d(128, sp_dim, 3, padding=1),
            torch.nn.BatchNorm1d(sp_dim),
            torch.nn.ReLU()
        )
        self.block_mid = spconv.SparseSequential(
            spconv.SparseConv3d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            spconv.SparseConv3d(in_channels=128, out_channels=sp_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(sp_dim),
            torch.nn.ReLU()
        )
        self.block_short = spconv.SparseSequential(
            spconv.SparseConv3d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            spconv.SparseConv3d(in_channels=128, out_channels=sp_dim, kernel_size=2, stride=2),
            torch.nn.BatchNorm1d(sp_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, x, types):
        if types == 'short':
            x = self.block_short(x)
        elif types == 'mid':
            x = self.block_mid(x)
        else:
            x = self.block_long(x)
        return x

# mean local feat
class SpconvNet(spconv.SparseModule):
    def __init__(self, sp_dim):
        super(SpconvNet, self).__init__()
        
        self.block = spconv.SparseSequential(
            spconv.SparseConv3d(3, 32, 3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            spconv.SparseConv3d(32, sp_dim, 3, padding=1),
            torch.nn.BatchNorm1d(sp_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x
'''