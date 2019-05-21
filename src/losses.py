# ref: https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/Losses.py
#!/usr/bin/python
import torch
import torch.nn as nn


class Loss():
    """Various loss function for GAN training."""

    def __init__(self, loss_type="hinge_loss"):
        """
        Loss function available:
            [hinge_loss, wass_loss, ls_loss, hinge_vae_loss]
        """
        loss_type = "_" + loss_type
        assert loss_type in self._losses_available(), f"Error, loss {loss_type} not implemented."

        self.disc_loss, self.gen_loss = getattr(self, loss_type)()

    def _losses_available(self):
        return [func for func in dir(self) if func.split("_")[-1] == "loss"]

    def _wass_loss(self):
        return _wass_disc_loss, _wass_gen_loss

    def _hinge_loss(self):
        return _hinge_disc_loss, _hinge_gen_loss

    def _ls_loss(self):
        return _ls_disc_loss, _ls_gen_loss

    def _hinge_vae_loss(self):
        return _hinge_disc_loss, _gen_hinge_loss_vae

########################
# Hinge Adversial loss #
########################


def _hinge_disc_loss(net_d, real_data, fake_data):
    # Train with real
    d_out_real = net_d(real_data)

    # Train with fake
    d_out_fake = net_d(fake_data)

    # adversial hinge loss
    d_loss_real = nn.ReLU()(1.0 - d_out_real).mean()
    d_loss_fake = nn.ReLU()(1.0 + d_out_fake).mean()
    d_loss = d_loss_real + d_loss_fake

    return d_loss


def _hinge_gen_loss(net_d, fake_data):
    loss = -net_d(fake_data).mean()

    return loss

####################
# Wasserstein loss #
####################

# Ref:
# https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py


def _gradient_penalty(net_d, real_data, fake_data, lambda_=10):
    bs, *_ = real_data.shape

    epsilon = torch.rand((bs, 1, 1, 1), device=real_data.device)

    # Create the new fake sample (x_hat)
    x_hat = epsilon * real_data + ((1 - epsilon) * fake_data)
    x_hat.requires_grad_(True)

    op = net_d(x_hat)

    # Backward pass to compute the gradient
    gradients, *_ = torch.autograd.grad(outputs=op,
                                        inputs=x_hat,
                                        grad_outputs=torch.ones_like(op),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)

    # Use the computed gradient to compute the penalty
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean() * lambda_

    return penalty


def _wass_disc_loss(net_d, real_data, fake_data, drift=1e-3, gp=True):
    """Wasserstein loss of the discriminator."""
    real_out = net_d(real_data)
    fake_out = net_d(fake_data)

    loss = fake_out.mean() - real_out.mean() + (drift * (real_out**2).mean())

    if gp:
        loss += _gradient_penalty(net_d, real_data, fake_data)

    return loss


def _wass_gen_loss(net_d, fake_data):
    """Wasserstein loss of the generator."""
    loss = -net_d(fake_data).mean()

    return loss

#####################
# Least square loss #
#####################


CRITERION = nn.MSELoss()


def _ls_disc_loss(net_d, real_data, fake_data):
    """Least square loss of the discriminator."""
    batch_size, *_ = real_data.shape
    real_labels = torch.full((batch_size,), 1., device=real_data.device)
    fake_labels = torch.full((batch_size,), 0., device=real_data.device)

    output_real = net_d(real_data)
    loss_real = CRITERION(output_real, real_labels)

    output_fake = net_d(fake_data)
    loss_fake = CRITERION(output_fake, fake_labels)

    return loss_real + loss_fake


def _ls_gen_loss(net_d, fake_data):
    """Least square loss of the generator."""
    batch_size, *_ = fake_data.shape
    labels = torch.full((batch_size,), 1., device=fake_data.device)

    output = net_d(fake_data)
    loss_g = CRITERION(output, labels)

    return loss_g

################################
# variational autoencoder loss #
################################

def _gen_hinge_loss_vae(netD, fake_data, img_g, mu, logvar):

    fake_grayscale = rgb2gray(fake_data.permute(0,2,3,1).cpu().detach().numpy())
    fake_grayscale = torch.from_numpy(fake_grayscale).to(device)

    img_g = img_g.squeeze(1)

    BCE = nn.MSELoss()(fake_grayscale, img_g)
    #BCE = fun.binary_cross_entropy()(fake_grayscale, img_g, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = -netD(fake_data).mean()

    vae_pen = 0.1 #this could be dynamic ? hmm

    return loss+vae_pen*(BCE+KLD)


# Example of use:
# real_labels = torch.full((batch_size,), real_label, device=device)
# fake_labels = torch.full((batch_size,), fake_label, device=device)
#
# in epochs:
# reals = images
# noise = noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
# fakes = netG(noise).detach()
#
# loss_d = ls_dis_loss(net_d, reals, fakes, real_labels, fake_labels)
#

# loss_g = ls_gen_loss(net_d, fakes, real_labels)
