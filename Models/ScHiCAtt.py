import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
# Global Attention definition
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass for self-attention layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height)
        Returns:
            torch.Tensor: Output tensor after applying self-attention mechanism.
        """
        m_batchsize, C, width, height = x.size()
        # Create query, key, and value matrices
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x N x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N (energy map)
        attention = self.softmax(energy)  # Apply softmax to normalize attention scores
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B x C x N

        # Multiply attention map with the value matrix
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(m_batchsize, C, width, height)  # Reshape to match original input dimensions

        # Apply learnable scaling factor gamma
        out = self.gamma * out + x
        return out

# Replace SelfAttention with GlobalAttention in the Cascading_Block and ScHiCAtt model

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)

class Residual_Block(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out) + x)
        return out

class Cascading_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.r1, self.r2, self.r3 = Residual_Block(channels), Residual_Block(channels), Residual_Block(channels)
        self.c1, self.c2, self.c3 = Basic_Block(channels * 2, channels), Basic_Block(channels * 3, channels), Basic_Block(channels * 4, channels)
        # Replacing SelfAttention with GlobalAttention
        self.attn = SelfAttention(channels)

    def forward(self, x):
        c0 = o0 = x
        b1 = self.r1(o0)
        c1, o1 = torch.cat([c0, b1], dim=1), self.c1(torch.cat([c0, b1], dim=1))
        b2 = self.r2(o1)
        c2, o2 = torch.cat([c1, b2], dim=1), self.c2(torch.cat([c1, b2], dim=1))
        b3 = self.r3(o2)
        c3, o3 = torch.cat([c2, b3], dim=1), self.c3(torch.cat([c2, b3], dim=1))
        return self.attn(o3)

class ScHiCAtt(nn.Module):
    def __init__(self, num_channels=64):
        super().__init__()
        self.entry = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1)
        self.cb1, self.cb2, self.cb3, self.cb4, self.cb5 = [Cascading_Block(num_channels) for _ in range(5)]
        self.cv1, self.cv2, self.cv3, self.cv4, self.cv5 = [nn.Conv2d(num_channels * i, num_channels, kernel_size=1) for i in range(2, 7)]
        self.exit = nn.Conv2d(num_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.entry(x)
        c0 = o0 = x
        b1, c1, o1 = self.cb1(o0), torch.cat([c0, self.cb1(o0)], dim=1), self.cv1(torch.cat([c0, self.cb1(o0)], dim=1))
        b2, c2, o2 = self.cb2(o1), torch.cat([c1, self.cb2(o1)], dim=1), self.cv2(torch.cat([c1, self.cb2(o1)], dim=1))
        b3, c3, o3 = self.cb3(o2), torch.cat([c2, self.cb3(o2)], dim=1), self.cv3(torch.cat([c2, self.cb3(o2)], dim=1))
        b4, c4, o4 = self.cb4(o3), torch.cat([c3, self.cb4(o3)], dim=1), self.cv4(torch.cat([c3, self.cb4(o3)], dim=1))
        b5, c5, o5 = self.cb5(o4), torch.cat([c4, self.cb5(o4)], dim=1), self.cv5(torch.cat([c4, self.cb5(o4)], dim=1))
        return self.exit(o5)
