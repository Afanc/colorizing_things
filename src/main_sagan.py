#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/python

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.optim import Adam

#import encoder as enc
import generator as gen
import discriminator as disc
import STL10GrayColor as STLGray
import utils as utls
import losses
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb


# In[2]:



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data
transform = transforms.Compose([transforms.Resize(128)])

# Load STL10 dataset
stl10_trainset = STLGray.STL10GrayColor(root="./data",
                              split='train',
                              download=True,
                              transform=transform)

#TODO
#train+unlabeled in split

#########################
# Test TODO:
# update in the same time the encoder and the generator
# reduce the learning rate after n epochs
# 


# In[ ]:


# Parameters
batch_size = 25
# z_dim = 256
params_loader = {
    'batch_size': batch_size,
    'shuffle': False
}

train_loader = DataLoader(stl10_trainset, **params_loader)


# In[4]:


netG = gen.GeneratorSeg(color_ch=2)
netD = disc.SADiscriminator(in_dim=2)

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


# In[ ]:


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

for epoch in range(n_epochs):
    print("epoch :", epoch)

    for idx, (img_g, img_c) in enumerate(train_loader):
        
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
              f"iter[{idx}/{len(train_loader)}], "
              f"d_out_real: {m_d_loss:.4f}, "
              f"g_out_fake: {m_g_loss:.4f}")
        
        if idx % 100 == 0:
            
            grayscale = torch.squeeze(img_g.detach())
            img_display = utls.convert_lab2rgb(grayscale,
                                               fakes.detach())
            vutils.save_image(img_display.detach(),
                              f'./l_{epoch}_epoch_{idx}.png',
                              normalize=True)
            
        # Release the gpu memory
        del fakes, g_loss
            
        torch.cuda.empty_cache()


# In[ ]:


load_old_state = False

# Create model
encoder = enc.Encoder(z_dim=z_dim)

generator = gen.Generator(z_dim=z_dim, init_depth=512)

discriminator = disc.Discriminator(max_depth=512)

if load_old_state:
    # Caution: I saved models with wrong name !!!!!!!!
    checkpoint = torch.load('_weights_8_iteration_600.pth')
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
else:
    generator.apply(utls.weights_init)
    discriminator.apply(utls.weights_init)
    
# Load model on GPU
encoder = encoder.to(device)
generator = generator.to(device)
discriminator = discriminator.to(device)


optimizer_params = {
    'lr': 0.0001,
    'betas':(0.5, 0.999),
    'weight_decay': 1e-4
}

enc_loss = nn.MSELoss()

if load_old_state:
    optimizer_e.load_state_dict(checkpoint['optimizer_e_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
else:
    optimizer_e = torch.optim.Adam(encoder.parameters(), **optimizer_params)
    optimizer_g = torch.optim.Adam(generator.parameters(), **optimizer_params)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), **optimizer_params)


# In[ ]:


print(encoder)
print(generator)
print(discriminator)


# In[ ]:





# In[ ]:


n_epochs = 50


for epoch in range(n_epochs):
    print("epoch :", epoch)

    for i, (img_g, img_c) in enumerate(train_loader):
        
        img_g = img_g.to(device)
        img_c = img_c.to(device)
# 
        bs, *_ = img_g.shape
        if bs != batch_size:
            continue


        #######################
        # Train Discriminator #
        #######################
        img_features = encoder(img_g)

        img_colorized = generator(img_features.detach())

        loss_d = losses.dis_loss(discriminator, img_c, img_colorized.detach())

        #bp
        discriminator.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        
        #######################
        # Train Generator #
        #######################
        
        #img_colorized = generator(img_features) #re attach ?
        
        loss_g = losses.gen_loss(discriminator, img_colorized)
        
        #bp
        generator.zero_grad()     
        loss_g.backward()
        optimizer_g.step()
        
        #######################
        # Train Encoder #
        #######################
        
        #TODO BETTER WAY/optimizing img_colorized without detach
        #img_features = encoder(img_g)

        img_colorized = generator(img_features)
        
        loss_e = enc_loss(img_colorized, img_c)
        
        #bp
        encoder.zero_grad()
        loss_e.backward()
        optimizer_e.step()
        
        #printing shit
        if (i%10 == 0) :
            pass
            #print("iteration ", i, "out of ", len(train_loader.dataset)//batch_size,
                  #"\terrD : ", round(loss_d.item(),3), "\terrG : ", round(loss_g.item(),3), "\terrE : ", round(loss_e.item(), 3))
        
        
        if i%100 == 0:
            
            encoder.eval()
            generator.eval()

            img_features = encoder(img_g)
            img_colorized = generator(img_features)
            img_display = utls.convert_lab2rgb(img_g, img_colorized.detach())

            encoder.train()
            generator.train()
            
            vutils.save_image(img_display,
                              f"/var/tmp/stu04/___epoch_{epoch}_iteration_{i}.png",
                              nrow=5,
                              normalize=True)
            print(">plotted shit")        

        lossD.append(loss_d.item())
        lossG.append(loss_g.item())
        lossE.append(loss_e.item())

        j += 1
        with open("all_losses.txt", "a+") as f :
            f.write(str(j)+"\t"+
                    str(round(lossD[-1],3))+"\t"+
                    str(round(lossG[-1],3))+"\t"+
                    str(round(lossE[-1],3))+"\n")

        
    
    


# In[ ]:


fig, axs = plt.subplots(2, figsize=(10,10))
fig.subplots_adjust(hspace=0.3)


axs[0].set_title("All Losses")
axs[0].set_xlabel("iterations")
axs[0].set_ylabel("Loss")
axs[0].plot(G_losses,label="G")
axs[0].plot(D_losses,label="D")
axs[0].legend()

axs[1].set_title("After 1000 iterations")
axs[1].set_xlabel("iterations")
axs[1].set_ylabel("Loss")
axs[1].plot(G_losses[1000:],label="G")
axs[1].plot(D_losses[1000:],label="D")
axs[1].legend()


# In[ ]:



