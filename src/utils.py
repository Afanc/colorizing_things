#!/usr/bin/python
import warnings
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from skimage import color
import STL10 as stl


def get_dataset(size=128):
    resize = transforms.Compose([Resize(size)])

    all_images = stl.STL10(root='../data',
                           split='train+unlabeled',
                           transform=resize,
                           target_transform=resize,
                           download=True)

    return all_images


def convert_lab2rgb(L, ab):
    """Convert the batch images from lab -> rgb."""

    # Normalize grayscale to rgb
    L = L*255-128
    # ab = ab*100

    # Add missing dim:
    # batch_size x img_size x img_size -> batch_size x 1 x img_size x img_size
    L.unsqueeze_(1)

    # Concatanet gray tensor with ab tensor
    colorized_img = torch.cat((L, ab), 1)

    # Change torch.Tensor to numpy array and permute the dimensions
    # bs x ch x h x w -> bs x h x w x ch
    reversed_img = colorized_img.cpu().double().detach().permute(0, 2, 3, 1).numpy()

    # Iterate over the images of the batch because the lab2rgb() function
    # take in paramters only a tuple (h x w x ch).
    for i in range(reversed_img.shape[0]):
        # Remove warning of bad color range.
        warnings.simplefilter("ignore")
        reversed_img[i, :, :, :] = color.lab2rgb(reversed_img[i, :, :, :])

    # Convert back the numpy array to torch.Tensor.
    # bs x h x w x ch -> bs x ch x h x w
    output = torch.from_numpy(reversed_img).float().permute(0, 3, 1, 2)

    return output.cuda()
