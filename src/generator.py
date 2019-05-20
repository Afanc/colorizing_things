#!/usr/bin/python

import os
import ssl
import torch.nn as nn
import torch
import torchvision.models as models

from CustomLayers import sn_conv2d, sn_convT2d, GenBlock, SelfAttention


if (not os.environ.get('PYTHONHTTPSVERIFY', '')
        and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


class GeneratorUNet(nn.Module):
    """Generator UNet.
    This class take a black and white image and generate the colors of
    the given image.
    """
    def __init__(self, in_dim=1, out_dim=3):
        super(GeneratorUNet, self).__init__()

        resnet = models.resnet34(pretrained=True)

        features = list(resnet.children())

        self.convert_bw_to_rgb = nn.Sequential(
            nn.Conv2d(in_dim, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.enc1 = nn.Sequential(*features[:4])
        self.enc2 = features[4]
        self.enc3 = features[5]
        self.enc4 = features[6]

        self.gen4 = nn.Sequential(
            *([features[7]] +
              [sn_conv2d(512, 256, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(True),
               # Quick fix
               sn_convT2d(256, 256, kernel_size=4, stride=2, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(True),
               sn_convT2d(256, 128, kernel_size=4, stride=2, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(True)
               ])
        )

        self.gen3 = GenBlock(256, 64, 2)
        self.gen2 = GenBlock(128, 64, 2, up=False)
        self.gen1 = nn.Sequential(
            sn_conv2d(128, 64, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            sn_convT2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.last = nn.Sequential(
            sn_convT2d(64, out_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.attention = SelfAttention(64)
        self.attention2 = SelfAttention(128)

    def forward(self, x):
        x = self.convert_bw_to_rgb(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        gen4 = self.gen4(enc4)
        gen3 = self.gen3(torch.cat([enc3, gen4], 1))
        gen2 = self.gen2(torch.cat([enc2, gen3], 1))
        attn1 = self.attention2(torch.cat([enc1, gen2], 1))
        gen1 = self.gen1(attn1)
        attn2 = self.attention(gen1)
        out = self.last(attn2)

        return out
