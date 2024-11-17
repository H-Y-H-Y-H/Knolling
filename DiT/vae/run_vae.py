import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse
import cv2
from vae import VAE
from PIL import Image

class DoubleImageDataset(Dataset):
    def __init__(self, root_dir, data_name, m2t_Flag, transform=None):
        self.data_name = data_name
        self.img_root_dir = root_dir + f'origin_images_{data_name}/'
        self.transform = transform
        self.data_num = data_num
        self.m2t_Flag = m2t_Flag

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        embed_id_0 = idx//12
        embed_id_1 = idx%12
        img_path = self.img_root_dir+ f'label_{embed_id_0}_{embed_id_1}.png'


        img = Image.open(img_path).convert("RGB")  # Ensure 3-channel images

        if self.transform:
            img = self.transform(img)
        return img

mode = 0
# "0" collect latents through encoder
M2T_FLAG = False

model_name = 'vae_V2_128'
checkpoint = torch.load(f"Models/{model_name}.pt", map_location="cpu")
data_root = "../../dataset/VAE_1020_obj4/"
data_name = 'before'


image_size = 128
batch_size = 128

data_num = 1000*12 #12*12*1000
# Randomly split the dataset with a fixed random seed for reproducibility


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((image_size,image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

# Wrap the dataset with the custom DoubleImageDataset to get two images per sample
dataset = DoubleImageDataset(root_dir=data_root, data_name=data_name, m2t_Flag=M2T_FLAG, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

syth_Encoder = VAE(channel_in=3,
              ch=2,
              blocks=(1,4,8,16,32),
              latent_channels=32,
              num_res_blocks=1,
              norm_type='bn').to(device)

syth_Encoder.load_state_dict(checkpoint['model_state_dict'])
syth_Encoder.eval()

latent_synth_list = [] # latent space of synthesized images
latent_space_list = [] # latent space of all dataset images
indices_list = []
dist_list = []
latent_list = []
count = 0
with torch.no_grad():
    with torch.cuda.amp.autocast():
        for i, test_images in enumerate(tqdm(data_loader, leave=False)):

            recon_img, mu, log_var = syth_Encoder(test_images.to(device))


            # recon_img2 = vae_net.run_decoder(mu)

            mu = mu.detach().cpu().numpy()
            # print(mu.shape) # (128, 32, 4, 4)

            # latent = mu.reshape(mu.shape[0], -1)

            latent_space_list.append(mu)
        data_pack = np.concatenate(latent_space_list)
        print(data_pack.shape)
        np.save(data_root+f'{data_name}_latent.npy',data_pack)

        # shape: (12000, 32, 4, 4)


