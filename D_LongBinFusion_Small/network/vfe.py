import time
import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_

import numpy as np
import torch_scatter

torch.manual_seed(0)

class KPConv(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels, KP_extent=0.1*1.5):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size + 1  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KP_extent = KP_extent

        # Initialize weights
        self.weights = nn.Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        center_data = np.zeros((3, self.K))

        num_kernel_points = int(kernel_size/(2*2))
        # kernel_points = (np.arange(num_kernel_points+2, dtype=np.float) / (num_kernel_points) - 0.5) * 0.1
        kernel_points = np.array([-0.1, -0.05, 0.05, 0.1]) / 2.0
                
        for x in range(num_kernel_points):
            for y in range(num_kernel_points):
                center_data[0, num_kernel_points*x+y] = kernel_points[x+1]
                center_data[1, num_kernel_points*x+y] = kernel_points[y+1]
                
        for z in range(num_kernel_points):
            center_data[:, num_kernel_points*num_kernel_points*z:num_kernel_points*num_kernel_points*(z+1)] = center_data[:, 0:num_kernel_points*num_kernel_points]            
            center_data[2, num_kernel_points*num_kernel_points*z:num_kernel_points*num_kernel_points*(z+1)] = kernel_points[z+1]

        center_data[2, :] = center_data[2, :]/2.0
        self.kernel_points = nn.Parameter(torch.tensor(center_data.T, dtype=torch.float32),
                         requires_grad=False)
        

        # Reset parameters
        self.reset_parameters()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        
        return
  

    def forward(self, s_pts, x, unq_inv):
        """
        :param q_pts:           next_pts. (num_next_points * 3)
        :param s_pts:           input_pts. (num_input_points * 3)        
        :param x:               Features (num_input_points * K)        
        """            
        s_pts.unsqueeze_(1)
        x = x.reshape(x.shape[0], 1, -1)
        neighbors = s_pts.expand(-1, self.K, 3)  
        
        # Apply offsets to kernel points [n_points, n_kpoints, dim]        
        deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]      

        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=-1)
        
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
        all_weights.unsqueeze_(2)        
        weighted_features = torch.matmul(all_weights, x)

        weighted_features = weighted_features.reshape(weighted_features.shape[0], -1)        
        weighted_sum_features = torch_scatter.scatter(weighted_features, unq_inv, dim=0)
        
        weighted_sum_features = weighted_sum_features.reshape(weighted_sum_features.shape[0], self.K, -1)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_sum_features = weighted_sum_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_sum_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)#, q_pts

class VoxelFeatureExtraction(nn.Module):
    def __init__(self, input_channels, output_channels):
        nn.Module.__init__(self)
        self.output_channel = output_channels
        self.linear = nn.Linear(input_channels, output_channels, bias = False)
        self.norm = nn.BatchNorm1d(output_channels)
        self.input_channels = input_channels

    def forward(self,inputs):      
        x = self.linear(inputs)
        x = self.norm(x)                      
        pointwise = F.relu(x)
        
        return pointwise

class VoxelFeatureExtractionKPconv(nn.Module):
    def __init__(self, input_channels, output_channels, n_centers=1, dimension=3):
        nn.Module.__init__(self)
        self.output_channel = output_channels
        self.input_channel = input_channels
        self.n_centers = n_centers

        self.kpconv = KPConv(self.n_centers, self.input_channel, self.output_channel)

        self.norm = nn.BatchNorm1d(output_channels)

    def forward(self, inputs, features, unq_inv):
        input_ = inputs.clone()
        x = self.kpconv(input_, features, unq_inv)
        x = x.reshape(-1, self.output_channel)
        x = self.norm(x)
        pointwise = F.relu(x)

        return pointwise