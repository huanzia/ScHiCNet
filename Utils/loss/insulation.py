# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import pdb
# import numpy as np
# import torch
#
# class computeInsulation(torch.nn.Module):
#     def __init__(self, window_radius=10, deriv_size=10):
#         super(computeInsulation, self).__init__()
#         self.window_radius = window_radius
#         self.deriv_size  = deriv_size
#         self.di_pool     = torch.nn.AvgPool2d(kernel_size=(2*window_radius+1), stride=1) #51
#         self.top_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
#         self.bottom_pool = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
#
#     def forward(self, x):
#         iv     = self.di_pool(x)
#         iv     = torch.diagonal(iv, dim1=2, dim2=3)
#         iv     = torch.log2(iv/torch.mean(iv))
#         top    = self.top_pool(iv[:,:,self.deriv_size:])
#         bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])
#         dv     = (top-bottom)
#         # left   = torch.cat([torch.zeros(dv.shape[0], dv.shape[1],2), dv], dim=2)
#         left = torch.cat([torch.zeros(dv.shape[0], dv.shape[1], 2, dtype=dv.dtype, device=dv.device), dv], dim=2)
#
#         # right  = torch.cat([dv, torch.zeros(dv.shape[0], dv.shape[1],2)], dim=2)
#         right = torch.cat([dv,torch.zeros(dv.shape[0], dv.shape[1], 2, dtype=dv.dtype, device=dv.device)], dim=2)
#
#         band   = ((left<0) == torch.ones_like(left)) * ((right>0) == torch.ones_like(right))
#         band   = band[:,:,2:-2]
#         boundaries = []
#         for i in range(0, band.shape[0]):
#             cur_bound = torch.where(band[i,0])[0]+self.window_radius+self.deriv_size
#             boundaries.append(cur_bound)
#         return iv, dv, boundaries
#
# class InsulationLoss(torch.nn.Module):
#     def __init__(self, window_radius=4, deriv_size=4): # here we modified the two parameters, both two origial parmaters are 10
#         super(InsulationLoss, self).__init__()
#         self.deriv_size     = deriv_size
#         self.window_radius  = window_radius
#         self.di_pool        = torch.nn.AvgPool2d(kernel_size=window_radius, stride=1) # Hout = (Hin - kerneal_size)/stride + 1 == Hin -3
#         self.top_pool       = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1) # Hout = (Hin - kerneal_size)/stride + 1 == Hin -3
#         self.bottom_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1) # Hout = (Hin - kerneal_size)/stride + 1 == Hin -3
#
#     def indivInsulation(self, x):
#         iv     = self.di_pool(x)
#         iv     = torch.diagonal(iv, dim1=2, dim2=3)
#         iv     = torch.log2(iv/torch.mean(iv))
#         top    = self.top_pool(iv[:,:,self.deriv_size:])
#         bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])
#         dv     = (top-bottom)
#         return dv
#
#     def forward(self, output, target):
#         out_dv = self.indivInsulation(output)
#         tar_dv = self.indivInsulation(target)
#         loss   = F.mse_loss(tar_dv, out_dv)
#         return loss
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

class computeInsulation(torch.nn.Module):
    """
    更稳的实现：
    - log2(iv/mean) 时加入 eps，避免 -inf/NaN；
    - 生成 zero padding 时显式指定 dtype/device；
    - 保持与原接口兼容：返回 (iv, dv, boundaries)。
    """
    def __init__(self, window_radius=10, deriv_size=10, eps=1e-6):
        super().__init__()
        self.window_radius = window_radius
        self.deriv_size = deriv_size
        self.eps = eps
        self.di_pool  = torch.nn.AvgPool2d(kernel_size=(2*window_radius+1), stride=1)
        self.top_pool = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.bottom_pool = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)

    def forward(self, x):
        """
        x: (B,1,H,W) on any device
        返回:
          iv: (B,1,L)    —— 归一化并 log 后的对角向量
          dv: (B,1,L-2d) —— 上下窗口差
          boundaries: list of 1D LongTensor（每个 batch 的边界索引）
        """
        # diagonal insulation
        iv = self.di_pool(x)                            # (B,1,*,*)
        iv = torch.diagonal(iv, dim1=2, dim2=3)         # (B,1,L)
        mean = iv.mean(dim=2, keepdim=True)
        iv = torch.log2((iv + self.eps) / (mean + self.eps))

        # directional (top-bottom)
        top    = self.top_pool(iv[:, :, self.deriv_size:])
        bottom = self.bottom_pool(iv[:, :, :-self.deriv_size])
        dv     = top - bottom                           # (B,1,L-2*deriv_size)

        # 寻找过零带（与原实现一致，但 device-safe）
        zeros_like_2 = torch.zeros(dv.shape[0], dv.shape[1], 2,
                                   dtype=dv.dtype, device=dv.device)
        left  = torch.cat([zeros_like_2, dv], dim=2)
        right = torch.cat([dv, zeros_like_2], dim=2)
        band  = ((left < 0) & (right > 0))[:, :, 2:-2]

        boundaries = []
        offset = self.window_radius + self.deriv_size
        for i in range(band.shape[0]):
            cur_bound = torch.where(band[i, 0])[0] + offset
            boundaries.append(cur_bound)

        return iv, dv, boundaries


class InsulationLoss(torch.nn.Module):
    """
    与上面 computeInsulation 一致的口径；加 eps，数值更稳。
    """
    def __init__(self, window_radius=4, deriv_size=4, eps=1e-6):
        super().__init__()
        self.deriv_size     = deriv_size
        self.window_radius  = window_radius
        self.eps = eps
        self.di_pool     = torch.nn.AvgPool2d(kernel_size=window_radius, stride=1)
        self.top_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.bottom_pool = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)

    def indivInsulation(self, x):
        iv = self.di_pool(x)
        iv = torch.diagonal(iv, dim1=2, dim2=3)
        mean = iv.mean(dim=2, keepdim=True)
        iv = torch.log2((iv + self.eps) / (mean + self.eps))
        top    = self.top_pool(iv[:, :, self.deriv_size:])
        bottom = self.bottom_pool(iv[:, :, :-self.deriv_size])
        dv     = top - bottom
        return dv

    def forward(self, output, target):
        out_dv = self.indivInsulation(output)
        tar_dv = self.indivInsulation(target)
        return F.mse_loss(tar_dv, out_dv)

