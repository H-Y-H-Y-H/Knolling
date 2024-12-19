from diffusers import AutoencoderKL
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from m2t_model import *
torch.manual_seed(0)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
model_name = 'vae_V10-M2T'
pretrained_model = 'Models/'+model_name+'.pt'

data_root = "C:/Users/yuhan/PycharmProjects/knolling_data/dataset/"
batch_size = 128
each_obj_num_data = 1000*24*2
data_num = each_obj_num_data*7
save_dir = "."
image_size =128
num_labels=12

# Transform for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

class DoubleImageDataset(Dataset):
    def __init__(self, root_dir, data_num, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_num = data_num

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        obj_num_id = idx//each_obj_num_data
        self.root_data_path = self.root_dir + f'VAE_1118_obj{obj_num_id + 2}/'
        idx = idx - obj_num_id * each_obj_num_data

        input_embed_id_0 = idx // (num_labels**2)
        input_embed_id_1 = (idx % (num_labels**2)) // num_labels

        label_embed_id_0 = idx // (num_labels**2)
        label_embed_id_1 = (idx %  (num_labels**2)) % num_labels

        input_img_path = self.root_data_path + f'/origin_images_before/label_{input_embed_id_0}_{input_embed_id_1}.png'
        label_img_path = self.root_data_path + f'/origin_images_after/label_{label_embed_id_0}_{label_embed_id_1}.png'

        input_img = Image.open(input_img_path).convert("RGB")  # Ensure 3-channel images
        target_img = Image.open(label_img_path).convert("RGB")  # Ensure 3-channel images

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img, label_embed_id_1

# Wrap the dataset with the custom DoubleImageDataset to get two images per sample
dataset = DoubleImageDataset(root_dir=data_root, data_num=data_num, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


vae_model = AutoencoderKL(
    in_channels=3,                   # Input image channels (e.g., RGB)
    out_channels=3,                  # Output image channels (e.g., RGB)
    latent_channels=4,               # Latent space channels
    block_out_channels=[32, 64, 128],  # Feature map channels for each downsample block
    down_block_types=[
        "DownEncoderBlock2D",        # First downsampling block
        "DownEncoderBlock2D",        # Second downsampling block
        "DownEncoderBlock2D"         # Third downsampling block
    ],
    up_block_types=[
        "UpDecoderBlock2D",          # First upsampling block
        "UpDecoderBlock2D",          # Second upsampling block
        "UpDecoderBlock2D"           # Third upsampling block
    ]
).to(device)
mlp_model = MLPModel(input_dim=4*32*32, label_num=num_labels, hidden_dim=1024, output_dim=(4,32,32), num_res_blocks=2).to(device)

vae_with_mlp = VAEWithMLP(vae_model, mlp_model).to(device)
checkpoint = torch.load(pretrained_model, map_location="cpu")

vae_with_mlp.load_state_dict(checkpoint['model_state_dict'])

import torchvision.utils as vutils
vae_with_mlp.eval()
latent_list = []
with torch.no_grad():
    for i, (input_images, label_images, label_id_all) in enumerate(val_loader):
        input_images, label_images, label_id_all = input_images.to(device), label_images.to(device), label_id_all.to(device)

        for sample_id in range(len(input_images)):
            image = input_images[sample_id].unsqueeze(0)
            results = []
            for label_id in range(12):
                label_id = torch.tensor(label_id).to(device).unsqueeze(0)
                reconstructed, latents = vae_with_mlp(image, label_id)
                results.append(reconstructed.cpu())
            results = torch.cat(results)
            # paddings = torch.zeros_like(image.cpu())
            img_cat = torch.cat((image.cpu(), results), 0).float()
            vutils.save_image(img_cat,
                              "%s/%s/%s_test.png" % (save_dir,
                                                        "Results",
                                                        "Results"),
                              normalize=True,nrow=13)
            break
        break







