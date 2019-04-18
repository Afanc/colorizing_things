import torch.nn as nn

class Generator(nn.Module):
    """
    Generator class: Take a random vector Z_dim and transform it in image.

    First iteration of the generator class. In this project the random vector
    Z_dim is a vector of characteristics given by the encoder.
    """

    # TODO: Create a dynamic generator with (z_dim, img_size, ncc).
    def __init__(self, Z_dim, ngf, ncc=2):
        """
        In:
            Z_dim(torch.Tensor): Random vector to transform in image.
            ngf(int): Number of channel in the internal layers of the generator.
            ncc(int): Number of channel of the output. (In this project ncc=2).
        """
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            # formula: (in-1)* stride - 2*padding + kernel_size

            # 1 -> (1-1)*1 - 2*0 + 4 = 4
            # in: (100 x 4 x 4)
            nn.ConvTranspose2d(Z_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (ncc) x 64 x 64
            nn.ConvTranspose2d(ngf, ncc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ncc) x 128 x 128
        )

    def forward(self, input_):
        output = self.layers(input_)

        return output
