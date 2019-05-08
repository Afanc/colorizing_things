# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Afanc/colorizing_things/blob/master/src/main.ipynb
"""

#!/usr/bin/python

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import encoder as enc
import generator as gen
import discriminator as disc
import STL10GrayColor as STLGray
import utils as utls
import losses
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device is ", device)

#data
transform = transforms.Compose([transforms.Resize(128)])#,

# Load STL10 dataset
stl10_trainset = STLGray.STL10GrayColor(root="./data",
                              split='train+unlabeled',
                              download=True,
                              transform=transform)

#TODO
#train+unlabeled in split

# Parameters
batch_size = 256
z_dim = 512
params_loader = {
    'batch_size': batch_size,
    'shuffle': False
}

train_loader = DataLoader(stl10_trainset, **params_loader)

#demultiplier = dem.Demultiplier()
#demultiplier = demultiplier.to(device)

encoder = enc.Encoder(z_dim=z_dim)
encoder = encoder.to(device)

generator = gen.Generator(z_dim=z_dim, init_depth=512)
generator.apply(utls.weights_init)
generator = generator.to(device)

discriminator = disc.Discriminator(max_depth=512)
discriminator.apply(utls.weights_init)
discriminator = discriminator.to(device)

optimizer_params = {
    'lr': 0.0001,
    'betas': (0.5, 0.999)
}

enc_loss = nn.MSELoss()

#optimizer_m = torch.optim.Adam(demultiplier.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_e = torch.optim.Adam(encoder.parameters(), **optimizer_params)
optimizer_g = torch.optim.Adam(generator.parameters(), **optimizer_params)
optimizer_d = torch.optim.Adam(discriminator.parameters(), **optimizer_params)

#print(encoder)
#print(generator)
#print(discriminator)

n_epochs = 1000

real_label = 1.
fake_label = 0.

real_labels = torch.full((batch_size,), real_label, device=device)
fake_labels = torch.full((batch_size,), fake_label, device=device)

criterion = nn.MSELoss()

lossD, lossG, lossE = [], [], []

j = 0
with open("all_losses.txt", "w+") as f :
    f.write("iteration\tlossD\tlossG\tlossE\n")

for epoch in range(n_epochs):
    print("epoch :", epoch)

    for i, (img_g, img_c) in enumerate(train_loader):

        img_g = img_g.to(device)
        img_c = img_c.to(device)

        bs, *_ = img_g.shape
        if bs != batch_size:
            continue

        #######################
        # Train Discriminator #
        #######################
        img_features = encoder(img_g)

        img_colorized = generator(img_features.detach())

        loss_d = losses.dis_loss(discriminator, img_c, img_colorized.detach())
        # print(loss_d)
        # loss_d = losses.ls_dis_loss(discriminator,
        #                             img_c,
        #                             img_colorized.detach(),
        #                             real_labels,
        #                             fake_labels,
        #                             criterion)
        # print(loss_d)
        #bp
        discriminator.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        #######################
        # Train Generator #
        #######################

        #img_colorized = generator(img_features) #re attach ?

        loss_g = losses.gen_loss(discriminator, img_colorized)
        # loss_g = losses.ls_gen_loss(discriminator,
        #                             img_colorized,
        #                             fake_labels,
        #                             criterion)
        #bp
        generator.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # print(list(generator.parameters()))

        #######################
        # Train Encoder #
        #######################

        #TODO BETTER WAY/optimizing img_colorized without detach
        img_features = encoder(img_g)

        img_colorized = generator(img_features)

        loss_e = enc_loss(img_colorized, img_c)

        #bp
        encoder.zero_grad()
        loss_e.backward()
        optimizer_e.step()

        #printing shit
        #if i%10 == 0 :
        #    pass
            #print("iteration ", i, "out of ", len(train_loader.dataset)//batch_size,
            #      "\terrD : ", round(loss_d.item(),3),
            #      "\terrG : ", round(loss_g.item(),3),
            #      "\terrE : ", round(loss_e.item(),3))


        if i%100 == 0:
            img_display = utls.convert_lab2rgb(img_g, img_colorized.detach())

            vutils.save_image(img_display,
                              f"___epoch_{epoch}.png",
                              nrow=5,
                              normalize=True)
            print(">plotted shit")

        lossD.append(loss_d.item())
        lossG.append(loss_g.item())
        lossE.append(loss_e.item())

        torch.save(generator.state_dict(), f'./_weights_G_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'./_weights_D_{epoch}.pth')
        torch.save(encoder.state_dict(), f'./_weights_E_{epoch}.pth')

        j += 1
        with open("all_losses.txt", "a+") as f :
            f.write(str(j)+"\t"+
                    str(round(lossD[-1],3))+"\t"+
                    str(round(lossG[-1],3))+"\t"+
                    str(round(lossE[-1],3))+"\n")


