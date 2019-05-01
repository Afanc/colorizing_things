#!/usr/bin/python

import torch
import torch.nn as nn

class Demultiplier(nn.Module):
    """converts 1 channel to 3"""
    # o = [(i + 2p -k)/s + 1]

    def __init__(self):
        super(Demultiplier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,3), padding=1))
    
    def forward(self, x):
        #print(torch.unsqueeze(x, 1).shape)
        x = torch.unsqueeze(x,1)
        out = self.conv(x)
        return out
 
