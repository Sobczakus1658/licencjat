import tqdm
import pickle
import torch
import PIL.Image
import dnnlib

def custom_generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=8, gridh=8, device=torch.device('cuda'),
    num_steps=18
):
    # Mocno wzorowana funkcja z example.py, ale z uproszczeniami
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net_data = pickle.load(f)
    net = net_data['ema'].to(device)
    net.eval()

    print(f'Generating {batch_size} images...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)

    class_labels = None
    if hasattr(net, 'label_dim') and net.label_dim > 0:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]


    timesteps = torch.linspace(net.num_timesteps-1, 0, num_steps, dtype=torch.long, device=device)
    z = latents


    for i, t in tqdm.tqdm(enumerate(timesteps), total=num_steps, unit='step'):
        t_idx = t.long()
        alpha_t = net.alphas[t_idx].reshape(-1, 1, 1, 1)
        sigma_t = net.sigmas[t_idx].reshape(-1, 1, 1, 1)
        eta_t = net.etas[t_idx].reshape(-1, 1, 1, 1)

        # W tym przypadku wytrenowaliśmy, żeby sieć przewidywała ε_t
        eps_pred = net(z, sigma_t, class_labels)

        noise = torch.randn_like(z) if t_idx > 0 else torch.zeros_like(z)
        z = alpha_t * z + sigma_t * torch.sqrt(1 - eta_t**2) * eps_pred + sigma_t * eta_t * noise

    # To pochodzi z generatora EDM
    img = (z * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(gridh, gridw, net.img_channels, net.img_resolution, net.img_resolution)
    img = img.permute(0, 3, 1, 4, 2).reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    img = img.cpu().numpy()
    PIL.Image.fromarray(img, 'RGB' if net.img_channels == 3 else 'L').save(dest_path)
    print(f'Saved image grid to "{dest_path}".')

def main():
    custom_generate_image_grid(
        network_pkl='network/network.pkl',
        dest_path='custom_samples.png',
        seed=42,
        gridw=8,
        gridh=8,
        device=torch.device('cuda'),
        num_steps=50,
    )

if __name__ == "__main__":
    main()