#!/usr/bin/python
import STL10 as stl
import torchvision.transforms as transforms
from torchvision.transforms import Resize, Grayscale


def get_dataset(size=128) : 

    resize = transforms.Compose([Resize(size)])
    
    all_images = stl.STL10(root='../data',
                           split='train+unlabeled', 
                           transform=resize,
                           target_transform=resize, 
                           download=True)

    return all_images
