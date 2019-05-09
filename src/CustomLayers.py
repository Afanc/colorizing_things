import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def flatten(x):
    bs, ch, width, height = x.shape

    return x.view(bs, -1, width*height)

def sn_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return spectral_norm(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   stride,
                                   padding))

def sn_convT2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
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

        layers = [
            sn_convT2d(in_channels, in_channels, kernel_size=2, stride=2),
            sn_conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
            sn_conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
        ] * (nb_conv_layers - 2)
        layers += [
            sn_conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.generate = nn.Sequential(*layers)

    def forward(self, x):
        return self.generate(x)
