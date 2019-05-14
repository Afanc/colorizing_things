import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def flatten(img):
    bs, _, width, height = img.shape

    return img.view(bs, -1, width*height)


def sn_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    """Add a spectral normalization around the conv layer."""
    return spectral_norm(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   stride,
                                   padding))


def sn_convT2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Add a spectral normalization around the convTranspose layer."""
    return spectral_norm(nn.ConvTranspose2d(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride,
                                            padding))


class SelfAttention(nn.Module):

    def __init__(self, ch_in, sq_fact=8):
        super(SelfAttention, self).__init__()

        self.ch_in = ch_in
        self.query = sn_conv2d(self.ch_in, self.ch_in//sq_fact, 1)
        self.key = sn_conv2d(self.ch_in, self.ch_in//sq_fact, 1)
        self.value = sn_conv2d(self.ch_in, self.ch_in, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_shape = x.shape

        proj_query = flatten(self.query(x)).permute(0, 2, 1)
        proj_key = flatten(self.key(x))
        proj_value = flatten(self.value(x))

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(*x_shape)
        out = self.gamma*out + x

        return out


class GenBlock(nn.Module):

    def __init__(self, in_channels, out_channels, nb_conv_layers):
        super(GenBlock, self).__init__()
        middle_channels = in_channels // 2

        self.generate = nn.Sequential(
            sn_conv2d(in_channels,
                      middle_channels,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(True),
            sn_convT2d(middle_channels,
                       out_channels,
                       kernel_size=4,
                       stride=2,
                       padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.generate(x)
