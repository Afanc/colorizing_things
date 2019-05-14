#!/usr/bin/python
import warnings
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import STL10

from skimage import color


def _get_transform(img_size, nb_channels=3):
    mean_std = [[0.5]*nb_channels, [0.5]*nb_channels]

    transformations = [transforms.Resize(img_size),
                       transforms.ToTensor(),
                       transforms.Normalize(*mean_std)]

    if nb_channels == 1:
        transformations.insert(1, transforms.Grayscale())

    return transforms.Compose(transformations)


def _load_dataset(folder, split, transform):
    params_dataset = {
        "root": folder,
        "download": True,
        "split": split,
        "transform": transform
    }

    return STL10(**params_dataset)


def get_datasetsSTL10(img_size=128, folder="./data", split="train+unlabeled"):
    """
    Load the STL10 dataset with the given parameters.
    Return 2 version of the STL10, the first one in colors and the second one
    grayscaled.
    """
    transform_c = _get_transform(img_size)
    transform_g = _get_transform(img_size, nb_channels=1)

    stl10_dtset_c = _load_dataset(folder, split, transform_c)
    stl10_dtset_g = _load_dataset(folder, split, transform_g)

    return stl10_dtset_c, stl10_dtset_g


def get_loadersSTL10(batch_size=6, img_size=128, folder="./data",
                     split="train+unlabeled"):
    params_loader = {
        'batch_size': batch_size,
        'shuffle': False
    }
    stl10_dtset_c, stl10_dtset_g = get_datasetsSTL10(img_size, folder, split)

    train_loader_c = DataLoader(stl10_dtset_c, **params_loader)
    train_loader_g = DataLoader(stl10_dtset_g, **params_loader)

    return train_loader_c, train_loader_g


def convert_lab2rgb(L, ab, using_save_image=True):
    """Convert the batch images from lab -> rgb."""

    # Normalize grayscale to rgb
    L = L*255-128
    ab *= 100

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
    if using_save_image:
        output = torch.from_numpy(reversed_img).float().permute(0, 3, 1, 2)
    else:
        output = torch.from_numpy(reversed_img).float().cuda() #.permute(0, 3, 1, 2)

    return output


def xavier_init_weights(model):
    """Init the weights of the given model with XAVIER."""
    if isinstance(model, (nn.Conv2d, nn.Linear)):
        xavier_uniform_(model.weight)
        model.bias.data.fill_(0.)


# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# custom weights initialization called on netG and netD
def weights_init(model):
    """Init the weights of the given model with normal distribution."""
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
