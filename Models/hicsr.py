import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 加 padding
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 加 padding
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res

class Generator(nn.Module):
    def __init__(self, num_res_blocks=5):
        super(Generator, self).__init__()

        self.pre_res_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 保持尺寸
            nn.ReLU()
        )

        res_blocks = [ResidualBlock(64) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.post_res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 保持尺寸
            nn.BatchNorm2d(64)
        )

        self.final_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        first_block = self.pre_res_block(x)
        res_blocks = self.res_blocks(first_block)
        post_res_block = self.post_res_block(res_blocks)
        final_block = self.final_block(first_block + post_res_block)
        return torch.tanh(final_block)

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 40 → 20
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 20 → 10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 10 → 5
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),  # 保持 5×5
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)  # 输出 1×5×5
        )

    def forward(self, x):
        return self.conv(x)

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)
