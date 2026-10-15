"""Student network: camera image -> [cx, cy, size, grip]. MobileNetV2 (ReLU6, depthwise convs: NPU-friendly) at 128x128."""
import torch
import torch.nn as nn
import torchvision


class HandNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        self.features = net.features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(1280, 4))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("t_mean", torch.zeros(4))      # target statistics, filled at training time
        self.register_buffer("t_std", torch.ones(4))

    def forward(self, x):
        """x: float32 RGB image, NCHW, values 0..255 (so the board can feed camera pixels directly). Returns real units."""
        z = self.head(self.features((x - self.mean) / self.std))
        return z * self.t_std + self.t_mean

    def forward_standardised(self, x):
        return self.head(self.features((x - self.mean) / self.std))
