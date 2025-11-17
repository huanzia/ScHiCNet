import torch
import torch.nn as nn
import torch.nn.functional as F
class SelectiveKernelFusion(nn.Module):
    def __init__(self, n_feats, reduction=8):
        super().__init__()
        self.conv3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(n_feats, n_feats, kernel_size=7, padding=3)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=False)
            for _ in range(3)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        feats = [x3, x5, x7]
        stack = torch.stack(feats, dim=1)  # B x 3 x C x H x W

        # 全局描述子
        U = sum(feats)
        s = self.global_pool(U)  # B x C x 1 x 1
        z = self.fc(s)           # B x C//reduction x 1 x 1

        # 为每个分支生成权重
        weights = [fc(z) for fc in self.fcs]  # List[B x C x 1 x 1]
        weights = torch.stack(weights, dim=1)  # B x 3 x C x 1 x 1
        weights = self.softmax(weights)       # B x 3 x C x 1 x 1

        # 融合
        out = (stack * weights).sum(dim=1)
        return out

# ---- Channel Attention ----
class ChannelAttention(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, 1),
            nn.GELU(),
            nn.Conv2d(n_feats // reduction, n_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return y

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        y = torch.cat([avg, max_], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)  # ✅ 只输出注意力图


# ---- Lightweight Self Attention ----
class LightSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, C, -1)           # B x C x HW
        k = self.key(x).view(B, C, -1)             # B x C x HW
        v = self.value(x).view(B, C, -1)           # B x C x HW

        attn = torch.bmm(q.transpose(1, 2), k) / (C ** 0.5)  # B x HW x HW
        attn = self.softmax(attn)
        out = torch.bmm(attn, v.transpose(1, 2))    # B x HW x C
        out = out.transpose(1, 2).view(B, C, H, W)
        return out + x

class EnhancedBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.ca = ChannelAttention(n_feats)
        self.sa = SpatialAttention()
        self.attn = LightSelfAttention(n_feats)
        self.res_scale = 0.1

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)

        ca_weight = self.ca(res)
        res = res * ca_weight  # 通道加权

        sa_weight = self.sa(res)
        res = res * sa_weight  # 空间加权

        res = self.attn(res)
        return x + res * self.res_scale

# ---- Residual Group ----
class ResidualGroup(nn.Module):
    def __init__(self, n_feats, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([EnhancedBlock(n_feats) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    def forward(self, x):
        residual = x
        for block in self.blocks:
            x = block(x)
        x = self.conv(x)
        return x + residual

class schicnet_Block(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_feats=64, num_groups=4, blocks_per_group=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_feats, 3, padding=1),
            nn.GELU(),
            SelectiveKernelFusion(n_feats)
        )

        self.body = nn.Sequential(*[ResidualGroup(n_feats, blocks_per_group) for _ in range(num_groups)])

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.Conv2d(n_feats, out_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        out = self.tail(res + x)
        return out
