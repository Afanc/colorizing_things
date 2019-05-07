# ref: https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/Losses.py

def gradient_penalty(netD, real_data, fake_data, lambda_=10):
    bs, *_ = real_data.shape

    epsilon = torch.rand((bs, 1, 1, 1), device=real_data.device)

    # Create the new fake sample (x_hat)
    x_hat = epsilon * real_data + ((1 - epsilon) * fake_data)
    x_hat.requires_grad_(True)

    op = netD(x_hat)

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

def dis_loss(netD, real_data, fake_data, drift=1e-3):
    """Compute the loss of the discriminator."""
    real_out = netD(real_data)
    fake_out = netD(fake_data)

    loss = fake_out.mean() - real_out.mean()\
           + (drift * (real_out**2).mean())

    loss += gradient_penalty(netD, real_data, fake_data)

    return Loss

def gen_loss(netD, fake_data):
    loss = -netD(fake_data).mean()

    return loss
