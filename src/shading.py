import torch.nn as nn
import torch
from CustomLayers import sn_conv2d, sn_convT2d, SelfAttention


class Shading(nn.Module):

    def __init__(self):
        super(Shading, self).__init__()

        self.enc1 = nn.Sequential(
            sn_conv2d(4, 8, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            sn_conv2d(8, 32, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.gen1 = nn.Sequential(
            sn_convT2d(32, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.gen2 = nn.Sequential(
            sn_convT2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            sn_conv2d(8, 3, padding=1),
            nn.Tanh()
        )
        # self.attn = SelfAttention(8)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        gen1 = self.gen1(enc2)
        gen2 = self.gen2(torch.cat([gen1, enc1], 1))

#        attn = self.attn(gen2)
#        out = self.out(attn)
        out = self.out(gen2)

        return out
