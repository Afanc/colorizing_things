#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.optim import Adam
import torchvision.utils as vutils

from discriminator import SADiscriminator
from generator import GeneratorUNet
from losses import Loss
from utils import get_loadersSTL10, xavier_init_weights, convert1


class Trainer():

    def __init__(self, batch_size=6, n_epochs=50, device=torch.device('cuda'),
                 lr_g=0.0001, lr_d=0.0004, betas=(0., 0.9), load_weights='',
                 loss_type="hinge_loss", folder_save="/var/tmp/stu04",
                 img_size=128):

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.folder_save = folder_save

        self.train_loader_c, self.train_loader_g = get_loadersSTL10(batch_size,
                                                                    img_size)
        self._init_models(load_weights)
        self._init_optimizers(lr_g, lr_d, betas)
        self.loss = Loss(loss_type)

        print("All packages loaded correctly.")

    def _init_models(self, load_weights):
        self.netG = GeneratorUNet()
        self.netD = SADiscriminator()

        if load_weights:
            checkpoint = torch.load(load_weights)

            self.netG.load_state_dict(checkpoint['generator_state_dict'])
            self.netD.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            self.netD.apply(xavier_init_weights)

        self.netG.to(self.device)
        self.netD.to(self.device)

    def _init_optimizers(self, lr_g, lr_d, betas):
        self.optimizer_g = Adam(self.netG.parameters(), lr=lr_g, betas=betas)
        self.optimizer_d = Adam(self.netD.parameters(), lr=lr_d, betas=betas)

    def _save_images(self, fakes, epoch, counter_iter, val="fakes"):
        vutils.save_image(
            fakes,
            f"{self.folder_save}/"
            f"X_{val}_epoch_{epoch}_iteration_{counter_iter}.png",
            nrow=6,
            normalize=True
        )

    def _save_models(self, epoch, counter_iter):
        torch.save({
            'generator_state_dict': self.netG.state_dict(),
            'discriminator_state_dict': self.netD.state_dict(),
        }, f'{self.folder_save}/X_weights_{epoch}_iteration_{counter_iter}.pth')

    def train(self):
        counter_iter = 0
        loaders = (self.train_loader_c, self.train_loader_g)

        with open("all_losses.txt", "w+") as file:
            file.write("iteration\tlossD\tlossG\n")

        losses_d = []
        losses_g = []

        for epoch in range(self.n_epochs):
            print("epoch :", epoch)

            # for idx, (img_g, img_c) in enumerate(train_loader):
            for idx, ((img_c, _), (img_g, _)) in enumerate(zip(*loaders)):
                img_g = img_g.to(self.device)
                img_c = img_c.to(self.device)

                # The last batch hasn't the same batch size so skip it
                bs, *_ = img_g.shape
                if bs != self.batch_size :
                    continue

                #######################
                # Train Discriminator #
                #######################

                # Create fake colors
                fakes = self.netG(img_g)

                d_loss = self.loss.disc_loss(self.netD,
                                             img_c,
                                             fakes.detach())

                m_d_loss = d_loss.item()

                # Backward and optimize
                self.netD.zero_grad()
                d_loss.backward()
                self.optimizer_d.step()

                # Release the gpu memory
                del d_loss

                ###################
                # Train Generator #
                ###################

                g_loss = self.loss.gen_loss(self.netD, fakes)

                # Backward and optimize
                self.netG.zero_grad()
                g_loss.backward()
                self.optimizer_g.step()

                m_g_loss = g_loss.item()

                if counter_iter % 100 == 0:
                    print(f"Epoch [{epoch}/{self.n_epochs}], "
                          f"iter[{idx}/{len(self.train_loader_g)}], "
                          f"d_out_real: {m_d_loss}, "
                          f"g_out_fake: {m_g_loss}")

                if counter_iter % 500 == 0:
                    self._save_images(fakes.detach(), epoch, counter_iter)

                    if counter_iter % 5000 == 0:
                        self._save_models(epoch, counter_iter)

                    print(">plotted and saved weights")

                # Release the gpu memory
                del fakes, g_loss

                losses_d.append(m_d_loss)
                losses_g.append(m_g_loss)

                torch.cuda.empty_cache()

                with open("all_losses.txt", "a+") as file:
                    file.write(str(counter_iter) + "\t" +
                               str(round(losses_d[-1], 3)) + "\t" +
                               str(round(losses_g[-1], 3)) + "\n")
                counter_iter += 1
