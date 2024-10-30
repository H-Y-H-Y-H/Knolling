# Import statements and initial setup
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from torchvision import transforms
from PIL import Image
import os
from tqdm.auto import tqdm

def main(args):
    # Setup PyTorch
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)

    # Extract model parameters from checkpoint
    ckpt_args = checkpoint['args']
    image_size = ckpt_args.image_size
    model_type = ckpt_args.model

    # Load model
    latent_size = image_size // 8
    model = DiT_models[model_type](input_size=latent_size).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # important!
    
    # Ensure the model has the forward_with_cfg_image method
    model.forward_with_cfg_image = model.forward_with_cfg_image.__get__(model)

    # Load diffusion and VAE
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True),
    ])

    # Load conditioning images
    n = args.num_samples
    if os.path.isfile(args.conditioning_path):
        # Single image
        conditioning_image = Image.open(args.conditioning_path).convert('RGB')
        conditioning_image = transform(conditioning_image).unsqueeze(0).to(device)
        conditioning_images = conditioning_image.repeat(n, 1, 1, 1)
    elif os.path.isdir(args.conditioning_path):
        # Load images from directory
        image_files = os.listdir(args.conditioning_path)
        image_files = image_files[:n]  # Limit to n images
        conditioning_images = []
        for img_file in image_files:
            img_path = os.path.join(args.conditioning_path, img_file)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            conditioning_images.append(img)
        conditioning_images = torch.stack(conditioning_images).to(device)
    else:
        raise ValueError("Invalid conditioning_path")

    if conditioning_images.shape[0] != n:
        raise ValueError("Number of conditioning images must be equal to num_samples")

    # Encode conditioning images
    with torch.no_grad():
        conditioning_latents = vae.encode(conditioning_images).latent_dist.sample() * 0.18215

    # Prepare initial noise z
    z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Implement classifier-free guidance
    z = torch.cat([z, z], 0)
    conditioning_latents = torch.cat([conditioning_latents, torch.zeros_like(conditioning_latents)], dim=0)

    # Prepare timesteps
    total_steps = diffusion.num_timesteps
    save_interval = max(1, total_steps // 10)
    indices = list(range(total_steps))[::-1]

    model_kwargs = {
        'conditioning_image': conditioning_latents,
    }

    # Sampling loop
    samples = None
    for step, out in enumerate(tqdm(
        diffusion.p_sample_loop_progressive(
            model.forward_with_cfg_image,
            shape=z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=model_kwargs,  # Pass model_kwargs here
            device=device,
        ),
        total=total_steps,
        desc="Sampling"
    )):
        current_timestep = indices[step]
        samples = out['sample']

        # Optionally save intermediate samples
        if current_timestep % save_interval == 0 or current_timestep == indices[-1]:
            samples_to_save = samples.clone()
            # samples_to_save = samples_to_save[:n]  # Take the first half (guided samples)
            samples_to_save = vae.decode(samples_to_save / 0.18215).sample
            save_image(
                samples_to_save,
                f"imgs/pretrain_sample_step_{current_timestep}.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"Intermediate samples at timestep {current_timestep} saved to sample_step_{current_timestep}.png")

    # Finalize and save the samples
    # samples = samples[:n]  # Take the first half (guided samples)
    samples = vae.decode(samples / 0.18215).sample

    save_image(samples, args.output, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Samples saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--conditioning-path", type=str, required=True, help="Path to conditioning images or image")
    parser.add_argument("--output", type=str, default="imgs/pretrain_sample.png", help="Output file name")
    args = parser.parse_args()
    main(args)
