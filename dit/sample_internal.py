# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint:
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    # Extract model parameters from checkpoint:
    ckpt_args = checkpoint['args']
    image_size = ckpt_args.image_size
    num_classes = ckpt_args.num_classes
    model_type = ckpt_args.model

    # Load model:
    latent_size = image_size // 8
    model = DiT_models[model_type](
        input_size=latent_size,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # important!
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Create sampling noise:
    n = args.num_samples
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (n,), device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = None
    total_steps = diffusion.num_timesteps
    save_interval = max(1, total_steps // 10)  
    indices = list(range(total_steps))[::-1]  

    from tqdm.auto import tqdm  

    for step, out in enumerate(tqdm(
        diffusion.p_sample_loop_progressive(
            model.forward_with_cfg,
            shape=z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=device,
        ),
        total=total_steps,
        desc="Sampling"
    )):
        current_timestep = indices[step]
        samples = out['sample']


        if current_timestep % save_interval == 0 or current_timestep == indices[-1]:
            samples_to_save = samples.clone()
            samples_to_save, _ = samples_to_save.chunk(2, dim=0)  
            samples_to_save = vae.decode(samples_to_save / 0.18215).sample
            save_image(
                samples_to_save,
                f"imgs/sample_step_{current_timestep}.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"Intermediate samples at timestep {current_timestep} saved to sample_step_{current_timestep}.png")

    samples, _ = samples.chunk(2, dim=0)  
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, args.output, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Samples saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output", type=str, default="sample.png", help="Output file name")
    args = parser.parse_args()
    main(args)
