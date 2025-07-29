"""Training and evaluation"""

from absl import app
from absl import flags
import click
import dnnlib
import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import time


def get_dataset_multi_host(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True
    )

    # print(f"Load dataset slice {slice}/{num_slices}, trainset length {len(train_ds)}, evalset length {len(eval_ds)}")
    return trainloader


def elbo(x_t, x_0, x_hat):
    mse = ((x_0 - x_hat) ** 2).mean().cpu()
    return mse

def compute_loss(
    net, loss_fn, statistics_dir, MAX_BATCH, num_steps, batch_size, device
):

    print(f"compute_loss")
    trainloader = get_dataset_multi_host(batch_size)

    timesteps = torch.arange(num_steps, dtype=torch.float64, device=device) / (num_steps - 1)
    elbos_lst = [0] * num_steps
    with torch.no_grad():
        for j, t in tqdm(enumerate(timesteps), desc="Computing elbos..."):
            time_start = time.time()
            for i, (batch, _) in enumerate(trainloader):
                if i >= MAX_BATCH:
                    break
                time_spent = time.time() - time_start
                print(f"Batch {i}/{MAX_BATCH}, {time_spent:.2f} s")
                train_batch = batch.to(device).float()
                # train_batch = train_batch.permute(0, 3, 1, 2)
                x = train_batch

                elbos_lst[j] += loss_fn(net=net, images=train_batch, labels=None, fixed_t=t, do_weight=False).mean()
            elbos_lst[j] = elbos_lst[j] / MAX_BATCH
            print(f"elbo[{j}]: {elbos_lst[j]}")
    elbos_lst = np.asarray(elbos_lst)
    # np.savez_compressed(os.path.join(statistics_dir, f"elbos_{r}.npz"), elbos=elbos_lst)
    np.savez_compressed(os.path.join(statistics_dir, f"elbos.npz"), elbos=elbos_lst)


def collect_loss(statistics_dir):
    print("Collecting loss...")
    elbo_lsts = []
    for file in os.listdir(statistics_dir):
        if file.startswith("elbos_"):
            elbo_lst = np.load(os.path.join(statistics_dir, file))["elbos"]
            elbo_lsts.append(elbo_lst)
    np.savez_compressed(os.path.join(statistics_dir, "elbo.npz"), l=np.mean(elbo_lsts, axis=0))

def compute_loss_dist(statistics_dir, **kwargs):
    opts = dnnlib.EasyDict(kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    os.makedirs(statistics_dir, exist_ok=True)

    loss_kwargs = dnnlib.EasyDict()

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        loss_kwargs.class_name = 'training.loss.VELoss'
    else:
        assert opts.precond == 'edm'
        loss_kwargs.class_name = 'training.loss.EDMLoss'

    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss

    with dnnlib.util.open_url(opts.network, verbose=True) as f:
        net = pickle.load(f)['ema'].to(device)

    if opts.dry_run == True:
        return

    compute_loss(net, loss_fn, statistics_dir, opts.n_batch, opts.n_timesteps, opts.batch_size, device)

def compute_discretization(statistics_dir, n_timesteps, max_target_steps):
    print("compute_discretization")
    losses = np.load(os.path.join(statistics_dir, 'elbos.npz'))['elbos']
    losses= np.append(losses, 0)[::-1] # Adding loss for T=0 and reversing.
    print(losses)

    dp_loss = np.ones((n_timesteps + 1, max_target_steps + 1)) * np.inf

    def step_loss(s, t):
        sigma_t = 1.0
        return losses[t] / sigma_t

    dp_loss[0][0] = 0

    for t in range(1, n_timesteps + 1):
        for k in range(1, max_target_steps + 1):
            for s in range(0, t):
                dp_loss[t][k] = min(dp_loss[t][k], dp_loss[s][k-1] + step_loss(s, t))
            
    discretization = [[] for _ in range(max_target_steps + 1)]
    print(discretization)

    for target_steps in range(1, max_target_steps + 1):
        t = n_timesteps
        for k in range(target_steps, 0, -1):
            discretization[target_steps].append(t)
            best = np.inf
            s = None
            for i in range(0, t):
                if dp_loss[i][k-1] + step_loss(i, t) < best:
                    best = dp_loss[i][k] + step_loss(i, t)
                    s = i
            assert s is not None
            t = s
        # Convert indices to step_indices used in sampling.
        discretization[target_steps] = list(reversed([i - 1 for i in discretization[target_steps]]))
        print(f'{target_steps=}, {discretization[target_steps]=}')

    step_indices = np.array(discretization, dtype=object)
    np.savez_compressed(os.path.join(statistics_dir, f"step_indices.npz"), step_indices=step_indices)
    

@click.command()
@click.option('--workdir',                 help='Work directory', metavar='PATH|URL',                       type=str, required=True)
@click.option('--network',                 help='Network pickle filename', metavar='PATH|URL',              type=str, required=True)
@click.option('--n_timesteps',             help='Number of timesteps', metavar='INT',                       type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--batch_size',              help='Batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--n_batch',                 help='Number of batches', metavar='INT',                         type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--precond',                 help='Preconditioning & loss function', metavar='vp|ve|edm',     type=click.Choice(['vp', 've', 'edm']), default='vp', show_default=True)
@click.option('--max_target_steps',        help='Max number of discretization steps in DP', metavar='INT',  type=click.IntRange(min=1), default=5, show_default=True)
@click.option('-n', '--dry_run',           help='Print computation options and exit', is_flag=True)
def main(**kwargs):
    """ Compute schedule maximizing training ELBO.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    opts = dnnlib.EasyDict(kwargs)

    workdir = opts.workdir

    network_name = opts.network.rsplit('/', 1)[-1]
    statistics_dir = os.path.join(
        workdir, "statistics", f"{network_name}__{opts.n_timesteps}_{opts.n_batch}_{opts.batch_size}"
    )

    # Skip computing loss if it's already done.
    if os.path.exists(statistics_dir):
        print("Losses already computed. Skipping...")
    else:
        compute_loss_dist(statistics_dir, **kwargs)

    compute_discretization(statistics_dir, opts.n_timesteps, opts.max_target_steps)
    

    # import torch.multiprocessing as mp

    # mp.set_start_method(method="spawn", force=True)
    # print("Spawning processes...")
    # processes_l = [
    #     mp.Process(
    #         target=compute_elbos,
    #         args=(
    #             framework,
    #             statistics_dir,
    #             opt.n_batch,
    #             opt.n_timesteps,
    #             opt.batch_size,
    #             num_gpus,
    #             device,
    #             i,
    #         ),
    #     )
    #     for i in range(num_gpus)
    # ]

    # [p.start() for p in processes_l]
    # [p.join() for p in processes_l]

    # collect_elbos(statistics_dir)

if __name__ == "__main__":
    main()
