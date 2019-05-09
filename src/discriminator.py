#!/usr/bin/python

from torch import nn
import numpy as np

# TODO: Same as the generator create dynamicly the network.
class Discriminator(nn.Module):
    """
    Discriminator class is used to discriminate the images given to the network.

    This discrininator must check if the given a, b colors are real or not.
    """
    def __init__(self, img_size=128, ncc=2, init_depth=64, max_depth=1024):
        """
        In:
            ndf(int): Number of channels in the internal layers.
            ncc(int): Number of color channels of the images given in parameters
                      (in this project ncc=2).
        """

        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.ncc = ncc
        self.init_depth = init_depth
        self.max_depth = max_depth

        self._create_layers()

    def _create_layers(self):
        internal_layers = []
        depth_in = self.ncc
        depth_out = self.init_depth

        # Assume img size power of 2!
        for _ in range(1, int(np.log2(self.img_size))-1):
            # Reduce the img size by 2 each iteration\n",
            internal_layers += self._init_block(depth_in, depth_out)
            depth_in = depth_out
            if not depth_out == self.max_depth:
                depth_out *= 2

        # Last layer different from the others : depth x 4 x 4 -> 1 x 1 x 1\n",
        internal_layers.append(nn.Conv2d(depth_in, 1, 4, 1, 0, bias=False))

        self.depth_in = depth_in
        self.depth_out = depth_out
        self.layers = nn.Sequential(*internal_layers)

    def _init_block(self, depth_in, depth_out):
        block = [nn.Conv2d(depth_in, depth_out, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(depth_out),
                 nn.LeakyReLU(0.2)]

        return block

    def forward(self, input_):
        output = self.layers(input_)

        return output.view(-1, 1).squeeze(1)
from custumLayers import sn_conv2d, SelfAttention

class SADiscriminator(nn.Module):
    """
    Discriminator with self attention layers
    """

    def __init__(self, in_dim=3, img_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.img_size = img_size
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
            SelfAttention(256),
            sn_conv2d(self.conv_dim*4, self.conv_dim*8, 4, 2, 1),
            nn.LeakyReLU(0.1),
            SelfAttention(512),
            nn.Conv2d(self.conv_dim*8, 1, 4)
        )

    def forward(self, x):
        out = self.layers(x)

        return out.squeeze()
