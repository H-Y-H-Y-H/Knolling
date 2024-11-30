from diffusers import AutoencoderKL
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
model_name = 'vae_V10-1'

data_root = "C:/Users/yuhan/PycharmProjects/knolling_data/dataset/"
batch_size = 512
each_obj_num_data = 1000*24 #*2
data_num = each_obj_num_data*7
save_dir = "."
image_size =128
data_name = 'after'

# Transform for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Dataset and DataLoader
class DoubleImageDataset(Dataset):
    def __init__(self, root_dir, data_name, transform=None):
        self.root_dir = root_dir
        self.data_name = data_name
        self.transform = transform
        self.data_num = data_num

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        obj_num_id = idx//each_obj_num_data

        self.root_data_path = self.root_dir + f'VAE_1118_obj{obj_num_id + 2}/origin_images_{self.data_name}/'
        idx = idx - obj_num_id * each_obj_num_data

        embed_id_0 = idx//12
        embed_id_1 = idx % 12
        img_path = self.root_data_path+ f'label_{embed_id_0}_{embed_id_1}.png'
        img = Image.open(img_path).convert("RGB")  # Ensure 3-channel images

        if self.transform:
            img = self.transform(img)
        return img

# Wrap the dataset with the custom DoubleImageDataset to get two images per sample
dataset = DoubleImageDataset(root_dir=data_root, data_name=data_name, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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

checkpoint = torch.load("Models/" + model_name + ".pt", map_location="cpu")
vae_model.load_state_dict(checkpoint['model_state_dict'])

# for i in range(2,11):
#     os.makedirs(f"C:/Users/yuhan/PycharmProjects/knolling_data/dataset/VAE_1118_obj{i}/latent_after",exist_ok=True)
#     os.makedirs(f"C:/Users/yuhan/PycharmProjects/knolling_data/dataset/VAE_1118_obj{i}/latent_before",exist_ok=True)

save_path = "C:/Users/yuhan/PycharmProjects/knolling_data/latent/"
os.makedirs(save_path,exist_ok=True)

latent_list = []
with torch.no_grad():
    for i, (images) in enumerate(data_loader):
        print(i)
        images = images.to(device)
        latents = vae_model.encode(images).latent_dist.sample()

        latents = latents.detach().cpu().numpy()
        latent_list.append(latents)

    latent_cat = torch.cat(latent_list)
    print(latent_cat.shape)

    np.save(save_path+f'{data_name}_latent.npy',latent_cat)

        # for j in range(len(images)):
        #     print(i,j)
        #     # print(img_path[j] + 'npy')
        #     np.save(img_path[j] + 'npy', latents)




