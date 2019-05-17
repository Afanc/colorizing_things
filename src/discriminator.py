#!/usr/bin/python

from torch import nn
from CustomLayers import sn_conv2d, SelfAttention


class DiscBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(DiscBlock, self).__init__()

        self.layers = nn.Sequential(
            sn_conv2d(in_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.1),
            sn_conv2d(out_dim,
                      out_dim,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.1),
        )

        def forward(self, x):
            out = self.layers(x)

            return out


class SADiscriminator(nn.Module):
    """
    Discriminator with self attention layers
    """

    def __init__(self, in_dim=3, conv_dim=64):
        super(SADiscriminator, self).__init__()
        self.in_dim = in_dim
        self.conv_dim = conv_dim

        self.layers = nn.Sequential(
                DiscBlock(self.in_dim, self.conv_dim),
                DiscBlock(self.conv_dim, self.conv_dim*2),
                DiscBlock(self.conv_dim*2, self.conv_dim*4),
                DiscBlock(self.conv_dim*4, self.conv_dim*4),
                sn_conv2d(self.conv_dim*4, self.conv_dim*8, 4, 2, 1),
                nn.LeakyReLU(0.1),
                SelfAttention(512),
                nn.Conv2d(self.conv_dim*8, 1, 4)
        )
#        self.layers = nn.Sequential(
#            sn_conv2d(self.in_dim, self.conv_dim, 4, 2, 1),
#            nn.BatchNorm2d(self.conv_dim),
#            nn.LeakyReLU(0.1),
#            sn_conv2d(self.conv_dim,
#                      self.conv_dim,
#                      kernel_size=3,
#                      padding=1),
#            nn.BatchNorm2d(self.conv_dim),
#            nn.LeakyReLU(0.1),
#            sn_conv2d(self.conv_dim, self.conv_dim*2, 4, 2, 1),
#            nn.BatchNorm2d(self.conv_dim*2),
#            nn.LeakyReLU(0.1),
#            sn_conv2d(self.conv_dim*2,
#                      self.conv_dim*2,
#                      kernel_size=3,
#                      padding=1),
#            nn.BatchNorm2d(self.conv_dim*2),
#            nn.LeakyReLU(0.1),
#            sn_conv2d(self.conv_dim*2, self.conv_dim*4, 4, 2, 1),
#            nn.BatchNorm2d(self.conv_dim*4),
#            nn.LeakyReLU(0.1),
#            sn_conv2d(self.conv_dim*4,
#                      self.conv_dim*4,
#                      kernel_size=3,
#                      padding=1),
#            nn.BatchNorm2d(self.conv_dim*4),
#            nn.LeakyReLU(0.1),
#            sn_conv2d(self.conv_dim*4, self.conv_dim*4, 4, 2, 1),
#            nn.BatchNorm2d(self.conv_dim*4),
#            nn.LeakyReLU(0.1),
#            sn_conv2d(self.conv_dim*4,
#                      self.conv_dim*4,
#                      kernel_size=3,
#                      padding=1),
#            nn.BatchNorm2d(self.conv_dim*4),
#            nn.LeakyReLU(0.1),
#            sn_conv2d(self.conv_dim*4, self.conv_dim*8, 4, 2, 1),
#            nn.LeakyReLU(0.1),
#            SelfAttention(512),
#            nn.Conv2d(self.conv_dim*8, 1, 4)
#        )

    def forward(self, x):
        out = self.layers(x)

        return out.squeeze()
