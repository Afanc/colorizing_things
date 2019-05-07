#!/usr/bin/python

import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    """converts 1 channel to 3"""
    # o = [(i + 2p -k)/s + 1]

    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        vgg_model = models.vgg16(pretrained=True)

        self.conv_1_to_3 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=3,
                      kernel_size=(3, 3),
                      padding=1)
            )

        self.vgg = nn.Sequential(*list(vgg_model.children())[:-2][0][:],
                                 # hardcoded, should be adaptive to image size
                                 nn.Conv2d(512, z_dim, (4, 4), 1, 0),
                                 nn.LeakyReLU(0.2))


    def forward(self, x):
        #print(torch.unsqueeze(x, 1).shape)
        x = torch.unsqueeze(x, 1)
        out = self.conv_1_to_3(x)
        out = self.vgg(out)
        return out
