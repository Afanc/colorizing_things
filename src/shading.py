import torch.nn as nn
import torch
from CustomLayers import sn_conv2d, sn_convT2d, SelfAttention


class Shading(nn.Module):

    def __init__(self):
        super(Shading, self).__init__()

        self.enc1 = nn.Sequential(
            sn_conv2d(4, 16, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            sn_conv2d(16, 64, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.gen1 = nn.Sequential(
            sn_convT2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.gen2 = nn.Sequential(
            sn_convT2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            sn_conv2d(16, 3, padding=1)
        )
        self.attn = SelfAttention(16)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        gen1 = self.gen1(enc2)
        gen2 = self.gen2(torch.cat([gen1, enc1], 1))

        attn = self.attn(gen2)
        out = self.out(attn)

        return out
