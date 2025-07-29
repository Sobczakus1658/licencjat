# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.


IMPLEMENTED_SOLVERS = [
    'euler', 'heun', 
    'dpm_solver_2', 'dpm_solver_3', 
    'simple_second_order', 
    'cox_matthews', 'krogstad', 'krogstad_2', 
    'ostermann'
]


def h_phi_1(h):
    return torch.expm1(h)


def h_phi_2(h):
    return (torch.expm1(h) / h) - 1


def h_phi_3(h):
    return (torch.expm1(h) / (h**2)) - (1 / h) - 0.5


def edm_to_dpm_denoiser(denoised, x, sigma):
    return (x - denoised) / sigma


def dpm_net(net, class_labels, x, alpha, sigma):
    denoised = net(x / alpha, sigma / alpha, class_labels).to(torch.float64)
    return edm_to_dpm_denoiser(denoised, x / alpha, sigma / alpha)


def euler_step(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    h = t_nxt - t_cur
    denoised = net(x_cur / alpha(t_cur), sigma(t_cur), class_labels).to(torch.float64)
    d_cur = (sigma_deriv(t_cur) / sigma(t_cur) + alpha_deriv(t_cur) / alpha(t_cur)) * x_cur - sigma_deriv(t_cur) * alpha(t_cur) / sigma(t_cur) * denoised
    x_next = x_cur + h * d_cur
    return x_next


def heun_step(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    h = t_nxt - t_cur
    denoised = net(x_cur / alpha(t_cur), sigma(t_cur), class_labels).to(torch.float64)
    d_cur = (sigma_deriv(t_cur) / sigma(t_cur) + alpha_deriv(t_cur) / alpha(t_cur)) * x_cur - sigma_deriv(t_cur) * alpha(t_cur) / sigma(t_cur) * denoised
    x_prime = x_cur + h * d_cur
    t_prime = t_cur + h

    denoised = net(x_prime / alpha(t_prime), sigma(t_prime), class_labels).to(torch.float64)
    d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + alpha_deriv(t_prime) / alpha(t_prime)) * x_prime - sigma_deriv(t_prime) * alpha(t_prime) / sigma(t_prime) * denoised
    x_next = x_cur + h * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# Algorithm 1
def simple_second_order_step(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    lam_cur = lam(t_cur)
    lam_nxt = lam(t_nxt)
    h = lam_nxt - lam_cur
    s = lam_inv(lam_cur + h / 2)

    f_1 = dpm_net(net, class_labels, x_cur, alpha(t_cur), sigma(t_cur))
    u = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * h_phi_1(h / 2) * f_1
    f_2 = dpm_net(net, class_labels, u, alpha(s), sigma(s))
    
    b_1 = h_phi_1(h) - 2 * h_phi_2(h)
    b_2 = 2 * h_phi_2(h)
    x_nxt = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * (b_1 * f_1 + b_2 * f_2)
    return x_nxt


def dpm_solver_2(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    lam_cur = lam(t_cur)
    lam_nxt = lam(t_nxt)
    h = lam_nxt - lam_cur
    s = lam_inv(lam_cur + h / 2)

    f_1 = dpm_net(net, class_labels, x_cur, alpha(t_cur), sigma(t_cur))
    u = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * h_phi_1(h / 2) * f_1
    f_2 = dpm_net(net, class_labels, u, alpha(s), sigma(s))
    
    x_nxt = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * h_phi_1(h) * f_2
    return x_nxt


def dpm_solver_3(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    r1 = 1 / 3
    r2 = 2 / 3

    lam_cur = lam(t_cur)
    lam_nxt = lam(t_nxt)
    h = lam_nxt - lam_cur
    s_1 = lam_inv(lam_cur + r1 * h)
    s_2 = lam_inv(lam_cur + r2 * h)

    f_1 = dpm_net(net, class_labels, x_cur, alpha(t_cur), sigma(t_cur))
    
    u_1 = (alpha(s_1) / alpha(t_cur)) * x_cur - sigma(s_1) * h_phi_1(r1 * h) * f_1
    f_2 = dpm_net(net, class_labels, u_1, alpha(s_1), sigma(s_1))
    D_1 = f_2 - f_1

    u_2 = (alpha(s_2) / alpha(t_cur)) * x_cur - sigma(s_2) * h_phi_1(r2 * h) * f_1 \
        - (sigma(s_2) * r2 / r1) * h_phi_2(r2 * h) * D_1
    f_3 = dpm_net(net, class_labels, u_2, alpha(s_2), sigma(s_2))
    D_2 = f_3 - f_1

    x_nxt = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * h_phi_1(h) * f_1 - (sigma(t_nxt) / r2) * h_phi_2(h) * D_2
    return x_nxt


# Algorithm 2
def cox_matthews_step(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    lam_cur = lam(t_cur)
    lam_nxt = lam(t_nxt)
    h = lam_nxt - lam_cur
    s = lam_inv(lam_cur + h / 2)

    f_1 = dpm_net(net, class_labels, x_cur, alpha(t_cur), sigma(t_cur))

    u_2 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * h_phi_1(h / 2) * f_1
    f_2 = dpm_net(net, class_labels, u_2, alpha(s), sigma(s))

    u_3 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * h_phi_1(h / 2) * f_2
    f_3 = dpm_net(net, class_labels, u_3, alpha(s), sigma(s))

    u_4 = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * (
        h_phi_1(h / 2) * (torch.expm1(h/2)) * f_1 +
        2 * h_phi_1(h / 2) * f_3
    )
    f_4 = dpm_net(net, class_labels, u_4, alpha(t_nxt), sigma(t_nxt))

    b_1 = h_phi_1(h) - 3 * h_phi_2(h) + 4 * h_phi_3(h)
    b_2 = b_3 = 2 * h_phi_2(h) - 4 * h_phi_3(h)
    b_4 = 4 * h_phi_3(h) - h_phi_2(h)

    x_nxt = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * (b_1 * f_1 + b_2 * f_2 + b_3 * f_3 + b_4 * f_4)
    return x_nxt


# Algorithm 3
def krogstad_step(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    lam_cur = lam(t_cur)
    lam_nxt = lam(t_nxt)
    h = lam_nxt - lam_cur
    s = lam_inv(lam_cur + h / 2)

    f_1 = dpm_net(net, class_labels, x_cur, alpha(t_cur), sigma(t_cur))

    u_2 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * h_phi_1(h / 2) * f_1
    f_2 = dpm_net(net, class_labels, u_2, alpha(s), sigma(s))
    D_2 = f_2 - f_1

    u_3 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * (
        h_phi_1(h / 2) * f_1 + 
        2 * h_phi_2(h / 2) * D_2
    )
    f_3 = dpm_net(net, class_labels, u_3, alpha(s), sigma(s))
    D_3 = f_3 - f_1

    u_4 = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * (
        h_phi_1(h) * f_1 +
        2 * h_phi_2(h) * D_3
    )
    f_4 = dpm_net(net, class_labels, u_4, alpha(t_nxt), sigma(t_nxt))

    b_1 = h_phi_1(h) - 3 * h_phi_2(h) + 4 * h_phi_3(h)
    b_2 = b_3 = 2 * h_phi_2(h) - 4 * h_phi_3(h)
    b_4 = 4 * h_phi_3(h) - h_phi_2(h)

    x_nxt = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * (b_1 * f_1 + b_2 * f_2 + b_3 * f_3 + b_4 * f_4)
    return x_nxt


def krogstad_step_2(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    lam_cur = lam(t_cur)
    lam_nxt = lam(t_nxt)
    h = lam_nxt - lam_cur
    s = lam_inv(lam_cur + h / 2)

    f_1 = dpm_net(net, class_labels, x_cur, alpha(t_cur), sigma(t_cur))

    u_2 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * h_phi_1(h / 2) * f_1
    f_2 = dpm_net(net, class_labels, u_2, alpha(s), sigma(s))

    u_3 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * h_phi_1(h / 2) * f_2
    f_3 = dpm_net(net, class_labels, u_3, alpha(s), sigma(s))

    u_4 = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * h_phi_1(h) * f_3
    f_4 = dpm_net(net, class_labels, u_4, alpha(t_nxt), sigma(t_nxt))

    b_1 = h_phi_1(h) - 3 * h_phi_2(h) + 4 * h_phi_3(h)
    b_2 = b_3 = 2 * h_phi_2(h) - 4 * h_phi_3(h)
    b_4 = 4 * h_phi_3(h) - h_phi_2(h)

    x_nxt = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * (b_1 * f_1 + b_2 * f_2 + b_3 * f_3 + b_4 * f_4)
    return x_nxt


# Algorithm 4
def ostermann(
    net, x_cur, sigma, alpha, alpha_deriv, sigma_deriv, lam, lam_inv,
    t_cur, t_nxt, class_labels
):
    lam_cur = lam(t_cur)
    lam_nxt = lam(t_nxt)
    h = lam_nxt - lam_cur
    s = lam_inv(lam_cur + h / 2)

    f_1 = dpm_net(net, class_labels, x_cur, alpha(t_cur), sigma(t_cur))

    u_2 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * h_phi_1(h / 2) * f_1
    f_2 = dpm_net(net, class_labels, u_2, alpha(s), sigma(s))
    D_2 = f_2 - f_1

    u_3 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * (
        h_phi_1(h / 2) * f_1 + 
        2 * h_phi_2(h / 2) * D_2
    )
    f_3 = dpm_net(net, class_labels, u_3, alpha(s), sigma(s))
    D_3 = f_3 - f_1

    u_4 = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * (
        h_phi_1(h) * f_1 +
        h_phi_2(h) * (D_2 + D_3)
    )
    f_4 = dpm_net(net, class_labels, u_4, alpha(t_nxt), sigma(t_nxt))
    D_4 = f_4 - f_1

    u_5 = (alpha(s) / alpha(t_cur)) * x_cur - sigma(s) * (
        h_phi_1(0.5 * h) * f_1 - \
        h_phi_2(-0.5 * h) * 0.5 * (2 * D_2 + 2 * D_3 - D_4) - \
        h_phi_3(-0.5 * h) * (-D_2 - D_3 + D_4) - \
        h_phi_2(-h) * 0.25 * (D_2 + D_3 - D_4) - \
        h_phi_3(-h) * (-D_2 - D_3 + D_4)
    )
    f_5 = dpm_net(net, class_labels, u_5, alpha(s), sigma(s))
    D_5 = f_5 - f_1

    x_nxt = (alpha(t_nxt) / alpha(t_cur)) * x_cur - sigma(t_nxt) * (
        h_phi_1(h) * f_1 + \
        h_phi_2(h) * (-D_4 + 4 * D_5) + \
        h_phi_3(h) * (4 * D_4 - 8 * D_5)
    )

    return x_nxt


# Returns solver function and NFEs per step
def get_solver_by_name(solver):
    return {
        "euler": (euler_step, 1),
        "heun": (heun_step, 2),
        "simple_second_order": (simple_second_order_step, 2),
        "dpm_solver_2": (dpm_solver_2, 2),
        "dpm_solver_3": (dpm_solver_3, 3),
        "cox_matthews": (cox_matthews_step, 4),
        "krogstad": (krogstad_step, 4),
        "krogstad_2": (krogstad_step_2, 4),
        "ostermann": (ostermann, 5)
    }[solver]


def get_solver_by_nfe(nfe):
    return {
        1: euler_step,
        2: dpm_solver_2,
        3: dpm_solver_3,
        4: krogstad_step,
    }[nfe]


# Scaling wywalone bo jest ściśle zależne od schedule teraz (potrzebujemy t_lambda)
def deterministic_ablation_sampler(
    net, latents, class_labels=None, nfe=18, 
    sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', 
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000
):
    assert solver in IMPLEMENTED_SOLVERS
    assert discretization in ['vp', 've', 'iddpm', 'edm', 'dpm']
    assert schedule in ['vp', 've', 'linear']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002, 'dpm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80, 'dpm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    solver_step, nfe_per_step = get_solver_by_name(solver)
    num_steps = (nfe + nfe_per_step - 2) // nfe_per_step + 1 # ceil((nfe - 1) / nfe_per_step) -> normal steps, last step is euler

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if schedule == 'vp':
        alpha = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        alpha_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (alpha(t) ** 3)
    else:
        alpha = lambda t: 1
        alpha_deriv = lambda t: 0

    # Obrzydliwy kod ale niech zostanie chwilowo
    dpm_sigma = lambda t: sigma(t) * alpha(t)
    lam = lambda t: torch.log(alpha(t) / dpm_sigma(t))
    used_sigma = sigma if solver in ['euler', 'heun'] else dpm_sigma
    
    if schedule == 'vp':
        lam_inv = lambda l: (
            (2 * torch.log(torch.exp(-2 * l) + 1)) /
            (torch.sqrt(vp_beta_min ** 2 + 2 * vp_beta_d * torch.log(torch.exp(-2 * l) + 1)) + vp_beta_min)
        )
    elif schedule == 've':
        lam_inv = lambda l: torch.exp(-2 * l)
    else:
        lam_inv = lambda l: torch.exp(-l) 

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    elif discretization == 'edm':
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    else:
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        # print("sigma_inv ", sigma_inv(sigma_steps))
        assert discretization == 'dpm'
        t_start = torch.tensor(sigma_inv(sigma_max))
        t_end = torch.tensor(sigma_inv(sigma_min))
        lambda_start = lam(t_start)
        lambda_end = lam(t_end)
        lambda_steps = lambda_start + step_indices / (num_steps - 1) * (lambda_end - lambda_start)
        # print("lambda ", lambda_steps.tolist())
        t_steps = lam_inv(lambda_steps)
        # print("t_steps ", t_steps.tolist())
        # print(f'{lambda_start=}, {lambda_end=}')
        assert t_steps[0] > t_steps[-1]

    if discretization != 'dpm':
        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(net.round_sigma(sigma_steps))

    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_nxt = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_nxt) * alpha(t_nxt))

    delta_nfe = nfe_per_step
    nfe_left = nfe
    for i, (t_cur, t_nxt) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        if i == num_steps - 1:
            delta_nfe = 1
            solver_step = euler_step
        elif nfe_left - 1 < delta_nfe:
            assert i == num_steps - 2
            delta_nfe = nfe_left - 1
            solver_step = get_solver_by_nfe(delta_nfe)

        nfe_left -= delta_nfe
        x_next = solver_step(
            net=net, x_cur=x_next,
            alpha=alpha, alpha_deriv=alpha_deriv,
            sigma=used_sigma, sigma_deriv=sigma_deriv,
            lam=lam, lam_inv=lam_inv,
            t_cur=t_cur, t_nxt=t_nxt, class_labels=class_labels
        )

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--nfe',                     help='Number of NFEs per evaluation', metavar='INT',                     type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(IMPLEMENTED_SOLVERS))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm', 'dpm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        images = deterministic_ablation_sampler(net, latents, class_labels, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
