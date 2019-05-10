#!/usr/bin/python

import torch.nn as nn
import numpy as np

import os, ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '')
    and getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

class Generator(nn.Module):
    """
    Generator class: Take a random vector Z_dim and transform it in image.

    First iteration of the generator class. In this project the random vector
    Z_dim is a vector of characteristics given by the encoder.
    """

    # TODO: Create a dynamic generator with (z_dim, img_size, ncc).
    def __init__(self, img_size=128, ncc=2, z_dim=512, init_depth=1024, min_depth=32):
        """
        In:
            Z_dim(torch.Tensor): Random vector to transform in image.
            ngf(int): Number of channel in the internal layers of the generator.
            ncc(int): Number of channel of the output. (In this project ncc=2).
        """
        super(Generator, self).__init__()

        super(Generator, self).__init__()

        self.img_size = img_size
        self.ncc = ncc
        self.z_dim = z_dim
        self.init_depth = init_depth
        self.min_depth = min_depth

        self._create_layers()

    def _create_layers(self):
        internal_layers = []
        depth_in = self.z_dim
        depth_out = self.init_depth

        first = True

        # Assume img size power of 2!
        for _ in range(1, int(np.log2(self.img_size))-1):
            # Augment the img size by 2 each iteration
            internal_layers += self._block(depth_in, depth_out, first)
            depth_in = depth_out

            if not depth_out == self.min_depth:
                depth_out //= 2

            first = False

        # Last layer different from the others
        last_layer = [nn.ConvTranspose2d(depth_in, self.ncc, 4, 2, 1, bias=False),
                      nn.Tanh()]
        internal_layers += last_layer

        self.layers = nn.Sequential(*internal_layers)

    def _block(self, depth_in, depth_out, first=False):
        val = (4, 2, 1)

        if first:
            val = (4, 2, 0)

        block = [nn.ConvTranspose2d(depth_in, depth_out, *val, bias=False),
                 nn.BatchNorm2d(depth_out),
                 nn.ReLU()]

        return block


    def forward(self, input_):
        output = self.layers(input_)

        return output

import torch
from CustomLayers import sn_conv2d, sn_convT2d, GenBlock, SelfAttention
import torchvision.models as models

class GeneratorSeg(nn.Module):
    def __init__(self, color_ch=2):
        super(GeneratorSeg, self).__init__()
        # TODO: check nb channels in vgg

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
            sn_conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            sn_conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            sn_convT2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.gen3 = GenBlock(512, 128, 2)
        self.gen2 = GenBlock(256, 64, 2)
        # self.gen1 = GenBlock(128, color_ch, 3)
        self.gen1 = nn.Sequential(
            sn_conv2d(128, 64, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.last = nn.Sequential(
            sn_convT2d(64, color_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        # self.attention1 = SelfAttention(128)
        self.attention = SelfAttention(64)

    def forward(self, x):
        x = self.convert_bw_to_rgb(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        gen4 = self.gen4(enc4)
        gen3 = self.gen3(torch.cat([enc3, gen4], 1))
        # gen3 = self.attention1(gen3)
        # Maybe should apply the attention layer on the enc
        # enc2 = self.attention1(enc2)
        gen2 = self.gen2(torch.cat([enc2, gen3], 1))
        # gen2 = self.attention2(gen2)
        # enc1 = self.attention2(enc1)
        gen1 = self.gen1(torch.cat([enc1, gen2], 1))
        attn = self.attention(gen1)
        out = self.last(attn)
        return out
