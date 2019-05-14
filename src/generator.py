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
    def __init__(self, out_dim=3):
        super(GeneratorUNet, self).__init__()

        vgg = models.vgg19_bn(pretrained=True)

        features = list(vgg.features.children())

        self.convert_bw_to_rgb = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.enc1 = nn.Sequential(*features[:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])

        self.gen4 = nn.Sequential(
            *(features[40:-1] +
              [sn_conv2d(512, 256, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(True),
               sn_convT2d(256, 256, kernel_size=4, stride=2, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(True)])
        )

        self.gen3 = GenBlock(512, 128, 2)
        self.gen2 = GenBlock(256, 64, 2)
        self.gen1 = nn.Sequential(
            sn_conv2d(128, 64, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.last = nn.Sequential(
            sn_convT2d(64, out_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.attention = SelfAttention(64)

    def forward(self, x):
        x = self.convert_bw_to_rgb(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        gen4 = self.gen4(enc4)
        gen3 = self.gen3(torch.cat([enc3, gen4], 1))
        gen2 = self.gen2(torch.cat([enc2, gen3], 1))
        gen1 = self.gen1(torch.cat([enc1, gen2], 1))
        attn = self.attention(gen1)
        out = self.last(attn)

        return out
