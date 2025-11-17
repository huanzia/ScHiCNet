import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from Utils.loss.SSIM import ssim

# ---- TV Loss ----
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = c * (h - 1) * w
        count_w = c * h * (w - 1)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

# ---- Edge-aware Loss (Sobel) ----
def gradient_loss(pred, target):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    gx_pred = F.conv2d(pred, sobel_x, padding=1)
    gx_target = F.conv2d(target, sobel_x, padding=1)
    return F.l1_loss(gx_pred, gx_target)

# ---- Generator Loss (Lite, 适配小尺寸图像) ----
class GeneratorLoss_v4(nn.Module):
    def __init__(self, device='cuda'):
        super(GeneratorLoss_v4, self).__init__()
        vgg = vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss(tv_loss_weight=1.0)
        self.device = device

    def forward(self, out_images, target_images):
        # Resize to 224 for perceptual loss
        out_resized = F.interpolate(out_images, size=(224, 224), mode='bilinear', align_corners=False)
        target_resized = F.interpolate(target_images, size=(224, 224), mode='bilinear', align_corners=False)

        # Feature Matching Loss (multi-layer perceptual)
        features_out = [self.vgg[:10](out_resized.repeat(1,3,1,1)),
                        self.vgg[:19](out_resized.repeat(1,3,1,1)),
                        self.vgg[:28](out_resized.repeat(1,3,1,1))]
        features_target = [self.vgg[:10](target_resized.repeat(1,3,1,1)),
                           self.vgg[:19](target_resized.repeat(1,3,1,1)),
                           self.vgg[:28](target_resized.repeat(1,3,1,1))]
        feature_loss = sum(F.mse_loss(f1, f2) for f1, f2 in zip(features_out, features_target))

        # MSE Loss
        image_loss = self.mse_loss(out_images, target_images)

        # SSIM Loss (适配小图像)
        ssim_loss = 1 - ssim(out_images, target_images)

        # Edge Loss
        edge_loss = gradient_loss(out_images, target_images)

        # TV Loss
        tv_loss = self.tv_loss(out_images)

        # Total Loss
        total_loss = (
            image_loss +
            0.01 * feature_loss +
            0.15 * ssim_loss +
            0.05 * edge_loss +
            1e-5 * tv_loss
        )

        return {
            'total': total_loss,
            'image_loss': image_loss,
            'feature_loss': feature_loss,
            'ssim_loss': ssim_loss,
            'edge_loss': edge_loss,
            'tv_loss': tv_loss
        }
