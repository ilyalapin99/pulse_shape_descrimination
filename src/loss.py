import torch.nn.functional as F
import torch

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):

    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss