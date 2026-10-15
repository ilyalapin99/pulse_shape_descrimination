import torch.nn.functional as F
import torch
import torch.nn as nn

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):

    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def vae_loss_function_log(recon_x, x, mu, logvar, beta=1.0, eps=1e-5):
    # 1. Зажимаем значения снизу на 0 и добавляем eps, чтобы log был стабильным
    recon_log = torch.log(torch.clamp(recon_x, min=0) + eps)
    x_log = torch.log(torch.clamp(x, min=0) + eps)
    
    # 2. MSE в логарифмическом масштабе (хвост получает такой же вес, как и пик)
    recon_loss = F.mse_loss(recon_log, x_log, reduction='mean')
    
    # 3. Твой стандартный аналитический KL-loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss