#!usr/bin/python

import torch
from torch.utils.data import DataLoader
from torchvision import models
import utils as ut

def encoder() :

    images = ut.get_dataset()    #returns all images, color and bw

    image_loader = DataLoader(dataset=images, batch_size=32, shuffle=True)

    encoder = models.vgg16_bn(pretrained=True)
    encoder.features[0].in_channels=1
    #modifier la sortie - jarter le fc

    print(encoder.features)

encoder()
