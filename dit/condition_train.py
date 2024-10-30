# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from download import find_model

import wandb
import torch.nn.functional as F

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def get_image_size(data_path):
    """
    Get the size of the first image in the dataset.
    """
    dataset = ImageFolder(data_path)
    first_image_path = dataset.samples[0][0]
    with Image.open(first_image_path) as img:
        return min(img.size)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def loss_fn(predicted, target):
    return F.mse_loss(predicted, target)

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

from torch.utils.data import Dataset
from PIL import Image
import os

class PairedImageDataset(Dataset):
    def __init__(self, before_dir, after_dir, transform=None):
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.transform = transform

        self.before_images = sorted(os.listdir(before_dir))
        self.after_images = sorted(os.listdir(after_dir))

        assert len(self.before_images) == len(self.after_images), "The number of images in before and after directories must be the same."

    def __len__(self):
        return len(self.before_images)

    def __getitem__(self, idx):
        before_path = os.path.join(self.before_dir, self.before_images[idx])
        after_path = os.path.join(self.after_dir, self.after_images[idx])

        before_image = Image.open(before_path).convert('RGB')
        after_image = Image.open(after_path).convert('RGB')

        if self.transform:
            before_image = self.transform(before_image)
            after_image = self.transform(after_image)

        return before_image, after_image


#################################################################################
#                                  Training Loop                                #
#################################################################################

def run_training(rank, world_size, args):
    setup(rank, world_size)
    
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup wandb
    if rank == 0:
        wandb.init(project="dit-training", config=args, mode="offline")


    # Get image size from dataset
    if args.image_size is None:
        args.image_size = get_image_size(args.data_path)
        if args.image_size % 8 != 0:
            args.image_size = (args.image_size // 8) * 8  # Ensure divisible by 8
            
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=None)

    
    # Load pretrained weights:
    if args.pretrained_path:
        logger.info(f"Loading pretrained weights from {args.pretrained_path}")
        state_dict = find_model(args.pretrained_path)
        model.load_state_dict(state_dict)
    
    if args.freeze_layers:
        logger.info("Freezing all layers except the final layer")
        for name, param in model.named_parameters():
            if 'final_layer' not in name:  
                param.requires_grad = False
                
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    ema.eval()
    
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    if args.freeze_layers:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = PairedImageDataset(
        before_dir=os.path.join(args.data_path, 'messy/images_before'),
        after_dir=os.path.join(args.data_path, 'tidy/images_after'),
        transform=transform
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Training loop:
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for before_image, after_image in loader:
            before_image = before_image.to(device)
            after_image = after_image.to(device)

            with torch.no_grad():
                latents = vae.encode(after_image).latent_dist.sample().mul_(0.18215)

            with torch.no_grad():
                conditioning = vae.encode(before_image).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps, (latents.shape[0],), device=device)
            
            model_kwargs = {'conditioning_image': conditioning}
            
            loss_dict = diffusion.training_losses(model, latents, t, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean() 
            # loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank == 0:
                    wandb.log({
                        "train_loss": avg_loss,
                        "steps_per_sec": steps_per_sec,
                        "global_step": train_steps
                    })
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    wandb.save(checkpoint_path)
                dist.barrier()

    model.eval()
    logger.info("Training completed!")
    cleanup()

def main(args):
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        if world_size > 1:
            mp.spawn(run_training, args=(world_size, args), nprocs=world_size)
        else:
            run_training(0, 1, args)
    else:
        print("CUDA is not available. Training on CPU is not supported.")
        return


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).   
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/conditioning")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    # parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=50_00)
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to pretrained model weights")
    parser.add_argument("--freeze-layers", action="store_true", help="Freeze all layers except the final layer")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate for finetuning")
    args = parser.parse_args()
    main(args)
