#!/usr/bin/python
import warnings
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from skimage import color
#import STL10 as stl
from torchvision.datasets import STL10
import STL10GrayColor as stl_gray


def get_dataset(size=128):
    transform = transforms.Compose([transforms.Resize(128)])

    stl10_trainset = stl_gray.STL10GrayColor(root="../data",
                                    split='train',
                                    download=True,
                                    transform=transform)

    return stl10_trainset


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
    if using_save_image :
        output = torch.from_numpy(reversed_img).float().permute(0, 3, 1, 2)
    else :
        output = torch.from_numpy(reversed_img).float().cuda() #.permute(0, 3, 1, 2)

    return output

def train(model, training_loader) :

    model.train()

    for iteration, images in enumerate(training_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        out = model(images)

        loss = loss_function(out, labels)

        averages += (np.argmax(out.detach(), axis=1) == labels).sum().item()
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("Training iteration ", iteration, "out of 42")

    accuracy = averages/len(training_loader.dataset)
    epoch_train_loss = np.mean(train_losses)

    return((epoch_train_loss, accuracy))

# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
