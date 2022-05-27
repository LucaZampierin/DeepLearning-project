import torch
import torch.nn as nn
import torch.nn.functional as F

class L0Loss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, denoised, target, tot_epochs, epoch):

        new_gamma = self.gamma * (tot_epochs - epoch)/tot_epochs
        loss = (abs(denoised - target) + 10**(-8))**new_gamma

        return torch.mean(loss.view(-1))

class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(HDRLoss, self).__init__()
        self._eps = eps


    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""

        denoised_nograd = denoised.detach().clone().requires_grad_(False)
        loss = ((denoised - target) ** 2) / (denoised_nograd + self._eps) ** 2
        return torch.mean(loss.view(-1))