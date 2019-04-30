#!/usr/bin/python

import torch.nn as nn

class Generator(nn.Module):
    """
    Generator class: Take a random vector Z_dim and transform it in image.

    First iteration of the generator class. In this project the random vector
    Z_dim is a vector of characteristics given by the encoder.
    """

    # TODO: Create a dynamic generator with (z_dim, img_size, ncc).
    def __init__(self, img_size, ncc=3, z_dim=100, init_depth=1024, min_depth=32):
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
