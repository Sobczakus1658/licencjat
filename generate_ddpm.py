import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from training.networks import DiscreteDDPPrecond

def custom_generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=8, gridh=8, device=torch.device('cuda'),  # default CPU, change if needed
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    with dnnlib.util.open_url(network_pkl) as f:
        net_data = pickle.load(f)
    print(type(net_data['ema']))
    return 
    net = net_data['ema']
    net = net.to(device)
    net.eval()

    # Latents and labels
    print(f'Generating {batch_size} images...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim > 0:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Clamp noise levels to network's sigma range
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Create timesteps for sampler
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Initialize z (latent)
    z = latents.to(torch.float64) * t_steps[0]

    # Sampling loop
    for i, (t_cur, t_next) in tqdm.tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), total=len(t_steps) - 1, unit='step'):
        t_idx_cur = net.sigma_to_t(t_cur)
        t_idx_next = net.sigma_to_t(t_next)

        alpha_cur = net.alphas[t_idx_cur]
        alpha_next = net.alphas[t_idx_next]
        sigma_cur = t_cur
        sigma_next = t_next

        eta_s = net.etas[t_idx_cur] if t_idx_cur < len(net.etas) else net.etas[-1]

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        z_hat = z + torch.sqrt(t_hat ** 2 - t_cur ** 2) * S_noise * torch.randn_like(z)

        # Prepare inputs for your model forward
        sigma_hat = t_hat.reshape(-1, 1, 1, 1).to(torch.float32)

        # Direct call to your custom model forward
        eps_pred = net(z_hat.to(torch.float32), sigma_hat, class_labels).to(torch.float64)

        term1 = (sigma_next * torch.sqrt(1 - eta_s**2) - (alpha_next / alpha_cur) * sigma_cur) * eps_pred
        term2 = (alpha_next / alpha_cur) * z_hat
        noise_term = sigma_next * eta_s * torch.randn_like(z_hat)

        z = term1 + term2 + noise_term

    # Convert latent to image grid
    print(f'Saving image grid to "{dest_path}"...')
    image = (z * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()

    PIL.Image.fromarray(image, 'RGB' if net.img_channels == 3 else 'L').save(dest_path)
    print('Done.')

def main():
    custom_generate_image_grid(
        network_pkl='network/network.pkl',
        dest_path='custom_samples.png',
        seed=42,
        gridw=8,
        gridh=8,
        device=torch.device('cpu'),  # change to 'cuda' if GPU available
        num_steps=50,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
    )

if __name__ == "__main__":
    main()
