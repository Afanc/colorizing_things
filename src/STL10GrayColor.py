#!/usr/bin/python
import numpy as np
import torch
from torchvision.datasets import STL10
from PIL import Image
from skimage import io, color
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class STL10GrayColor(STL10):

    def __getitem__(self, index):
        """
        Overwrite the original method to return a grayscaled image (L) and
        the colors of the image (ab). The color space is CIE Lab.

        Input:
            index(int): index of the desired image

        Return:
            L_tensor(torch.Tensor): L dimension of the picture
            ab_tensor(torch.Tensor): ab dimension of the picture
        """
        img = self.data[index]

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

            # Convert img RGB -> Lab
            lab_img = color.rgb2lab(img)

            # Transpose the lab img: dim 128x128x3 -> 3x128x128
            L, a, b = np.transpose(lab_img, (2, 0, 1))

            # Normalize
            L_normalized = (L + 128) / 255
            ab_normalized = np.asarray([a, b], dtype=np.float32)

            # re-Transpose 3x128x128-> 128x128x3
            #L, a, b = np.transpose(lab_img, (1, 2, 0))

            # Transform numpy array to torch tensor
            L_tensor = torch.from_numpy(L_normalized.astype(np.float32))
            ab_tensor = torch.from_numpy(ab_normalized.astype(np.float32))

        return (L_tensor, ab_tensor)

if __name__ == "__main__":
    # Example of usage:

    # Image preprocessing
    transform = transforms.Compose([transforms.Resize(128)])#, transforms.ToTensor()])

    # Load STL10 dataset
    stl10_trainset = STL10GrayColor(root="./data",
                                    split='train',
                                    download=True,
                                    transform=transform)


    # Parameters
    params_loader = {'batch_size': 32,
                     'shuffle': False}

    train_loader = DataLoader(stl10_trainset, **params_loader)

    for idx, (image_g, image_c) in enumerate(train_loader) :
        print(type(image_g))
