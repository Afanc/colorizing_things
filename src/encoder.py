#!usr/bin/python

import torch
from torch.utils.data import DataLoader
from torchvision import models
import utils as ut
import torch.nn as nn

def Encoder() :
    """
    Encoder : returns a shortened version of the vgg16 pretrained model.
    """

    vgg_model = models.vgg16(pretrained=True)

    encoder = nn.Sequential(*list(vgg_model.children())[:-2][0][:],
                            nn.Conv2d(512, 512, (4,4), 1, 0),   #hardcoded, should be adaptive to image size
                            nn.LeakyReLU(0.2))
    #print(encoder)
    #encoder[0].in_channels=1
    
    #encoder = models.vgg16_bn(pretrained=True)
    #encoder.features[0].in_channels=1
    #encoder.features[43] = Identity()
    
    return encoder

if __name__ == "__main__" :
    e = Encoder()
