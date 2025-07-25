# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
@persistence.persistent_class


class DiscreteDDPLoss:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data
        self.alphas = None
        self.sigmas = None
        self.Ut = None 

    def setup_schedules(self, net):
        """Initialize schedules from network"""
        net_model = net.module if hasattr(net, 'module') else net
        with torch.no_grad():
            self.alphas = net_model.alphas
            self.sigmas = net_model.sigmas
            if hasattr(net, 'Ut'):
                self.Ut = net.Ut

    def __call__(self, net, images, labels=None, augment_pipe=None):
        if self.alphas is None:
            self.setup_schedules(net)

        # Generowanie z rozkładu jednostajnego
        t = torch.randint(1, len(self.alphas), (images.shape[0],), device=images.device)
        alpha_t, sigma_t = self.alphas[t], self.sigmas[t]
        alpha_s, sigma_s = self.alphas[t-1], self.sigmas[t-1]
    
        # Obliczanie współczynników
        # chwilowo wstawienie U_t jako 1 jako stała
        gamma_s = (alpha_s/sigma_s)**2 * (sigma_t/alpha_t)**2

        eta_s = (gamma_s - 1)/(torch.sqrt(gamma_s) + torch.sqrt(gamma_s - 1))
        eta_s = eta_s.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        eta_s = eta_s.view(-1,1,1,1)

        # Generowanie szumu
        eps_t = torch.randn_like(images)
        eps_s = torch.randn_like(images)

        epsilon_t = eps_t
        print(f"eta_s min/max: {eta_s.shape}")
        print(f"epsilon_t shape: {eps_t.shape}")
        
        epsilon_s = torch.sqrt(1 - eta_s**2) * epsilon_t + eta_s * eps_s

        # Obliczanie z_t i z_s
        z_t = alpha_t.view(-1,1,1,1)*images + sigma_t.view(-1,1,1,1)*epsilon_t
        z_s = alpha_s.view(-1,1,1,1)*images + sigma_s.view(-1,1,1,1)*epsilon_s

        with torch.no_grad():
            u_target = (z_s - alpha_s.view(-1,1,1,1)*images)/(sigma_s.view(-1,1,1,1))
        u_pred = net(z_t, sigma_t, labels)

        # Simplified loss (Eq. 16)
        loss = 0.5 * ((u_target - u_pred)**2).mean()
        
        return loss
