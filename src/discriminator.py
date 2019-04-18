from torch import nn

# TODO: Same as the generator create dynamicly the network.
class Discriminator(nn.Module):
    """
    Discriminator class is used to discriminate the images given to the network.

    This discrininator must check if the given a, b colors are real or not.
    """
    def __init__(self, ndf, ncc):
        """
        In:
            ndf(int): Number of channels in the internal layers.
            ncc(int): Number of color channels of the images given in parameters
                      (in this project ncc=2).
        """
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1, padding=0,
            # Formula: (H + 2*padding -kernel_size)/stride + 1

            # in: 3 x 128 x 128
            # out: 128 x 65 x 65 (64 x 33 x 33)
            nn.Conv2d(ncc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            #out: 256 x 33 x 33 (128 x 17 x 17)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),

            # out 512 x 17 x 17 (256 x 9 x 9)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            # out 1024 x 9 x 9 (512 x 5 x 5)
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2),

            # # out: 1024 x 5 x 5
            nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2),

            # out (1 x 1 x 1)
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()

        )

    def forward(self, input_):
        output = self.layers(input_)

        return output.view(-1, 1).squeeze(1)
