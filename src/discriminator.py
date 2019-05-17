#!/usr/bin/python

from torch import nn
from CustomLayers import sn_conv2d, SelfAttention


class SADiscriminator(nn.Module):
    """
    Discriminator with self attention layers
    """

    def __init__(self, in_dim=3, conv_dim=64):
        super(SADiscriminator, self).__init__()
        self.in_dim = in_dim
        self.conv_dim = conv_dim

        self.layers = nn.Sequential(
            sn_conv2d(self.in_dim, self.conv_dim, 4, 2, 1),
            nn.LeakyReLU(0.1),
            sn_conv2d(self.conv_dim, self.conv_dim*2, 4, 2, 1),
            nn.LeakyReLU(0.1),
            sn_conv2d(self.conv_dim*2, self.conv_dim*4, 4, 2, 1),
            nn.LeakyReLU(0.1),
            sn_conv2d(self.conv_dim*4, self.conv_dim*4, 4, 2, 1),
            nn.LeakyReLU(0.1),
            sn_conv2d(self.conv_dim*4, self.conv_dim*8, 4, 2, 1),
            nn.LeakyReLU(0.1),
            SelfAttention(512),
            nn.Conv2d(self.conv_dim*8, 1, 4)
        )

    def forward(self, x):
        out = self.layers(x)

        return out.squeeze()
