# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W970PbWuy9axDI-kl6JvTnCB0BjfrMH5
"""

#!/usr/bin/python

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.datasets as dsets

#import encoder as enc
import generator as gen
import discriminator as disc
import STL10GrayColor as STLGray
import utils as utls
import losses
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data
# transform = transforms.Compose([transforms.Resize(128)])
#
# # Load STL10 dataset
# stl10_trainset = STLGray.STL10GrayColor(root="./data",
#                               split='train',
#                               download=True,
#                               transform=transform)

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

stl10_dtset_c = dsets.STL10(root="./data",
                          download=True,
                          split='train+unlabeled',
                          transform=transform)
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ),
                         (0.5, )),
])
stl10_dtset_g = dsets.STL10(root="./data",
                          download=True,
                          split='train+unlabeled',
                          transform=transform)

#TODO
#train+unlabeled in split

#########################
# Test TODO:
# update in the same time the encoder and the generator
# reduce the learning rate after n epochs
#

# Parameters
batch_size = 6
# z_dim = 256
params_loader = {
    'batch_size': batch_size,
    'shuffle': False
}

train_loader_c = DataLoader(stl10_dtset_c, **params_loader)
train_loader_g = DataLoader(stl10_dtset_g, **params_loader)

netG = gen.GeneratorSeg()
netD = disc.SADiscriminator()

# TODO init layers of the generator in the class
netD.apply(utls.xavier_init_weights)

netG.to(device)
netD.to(device)

# parameters given in the original paper
lr_g = 0.0001
lr_d = 0.0004

betas = (0., 0.9)

optimizer_g = Adam(netG.parameters(), lr=lr_g, betas=betas)
optimizer_d = Adam(netD.parameters(), lr=lr_d, betas=betas)

print(netG)
print(netD)

n_epochs = 50
wass_loss = False

def disc_hinge_loss(netD, real_data, fake_data):
    # Train with real
    d_out_real = netD(real_data)

    # Train with fake
    d_out_fake = netD(fake_data)

    # adversial hinge loss
    d_loss_real = nn.ReLU()(1.0 - d_out_real).mean()
    d_loss_fake = nn.ReLU()(1.0 + d_out_fake).mean()
    d_loss = d_loss_real + d_loss_fake

    return d_loss

def gen_hinge_loss(netD, fake_data):
    loss = -netD(fake_data).mean()

    return loss

j = 0
with open("all_losses.txt", "w+") as f :
    f.write("iteration\tlossD\tlossG\n")

lossD = []
lossG = []

for epoch in range(n_epochs):
    print("epoch :", epoch)

    # for idx, (img_g, img_c) in enumerate(train_loader):
    for idx, ((img_c, _), (img_g, _)) in enumerate(zip(train_loader_c, train_loader_g)):
        img_g = img_g.to(device)
        img_c = img_c.to(device)

        # The last batch hasn't the same batch size so skip it
        bs, *_ = img_g.shape
        if bs != batch_size:
            continue

        #######################
        # Train Discriminator #
        #######################

        # Create fake colors
        fakes = netG(img_g)

        if wass_loss:
            d_loss = losses.dis_loss(netD, img_c, fakes.detach())
        else:
            d_loss = disc_hinge_loss(netD, img_c, fakes.detach())

        m_d_loss = d_loss.item()

        # Backward and optimize
        netD.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Release the gpu memory
        del d_loss

        #######################
        # Train Discriminator #
        #######################

        if wass_loss:
            g_loss = losses.gen_loss(netD, fakes)
        else:
            g_loss = gen_hinge_loss(netD, fakes)

        # Backward and optimize
        netG.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        m_g_loss = g_loss.item()


        print(f"Epoch [{epoch}/{n_epochs}], "
              f"iter[{idx}/{len(train_loader_g)}], "
              f"d_out_real: {m_d_loss}, "
              f"g_out_fake: {m_g_loss}")

        if j%100 == 0:

            netG.eval()
            fakes = netG(img_g)
            #img_features = encoder(img_g)
            #img_colorized = generator(img_features)
            #img_display = utls.convert_lab2rgb(img_g, img_colorized.detach())

            netG.train()

            vutils.save_image(fakes,
                              f"/var/tmp/stu04/___epoch_{epoch}_iteration_{j}.png",
                              nrow=5,
                              normalize=True)

            if j%5000 == 0:
                torch.save({
                    'generator_state_dict': netG.state_dict(),
                    'discriminator_state_dict': netD.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                }, f'/var/tmp/stu04/_weights_{epoch}_iteration_{j}.pth')

            print(">plotted and saved weights")

        lossD.append(m_d_loss)
        lossG.append(m_g_loss)


        # Release the gpu memory
        del fakes, g_loss

        torch.cuda.empty_cache()

        with open("all_losses.txt", "a+") as f :
            f.write(str(j)+"\t"+
                    str(round(lossD[-1],3))+"\t"+
                    str(round(lossG[-1],3))+"\n")
        j += 1
