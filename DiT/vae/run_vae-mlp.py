import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as T
from vgg19 import VGG19
from torch.optim.lr_scheduler import ReduceLROnPlateau
from vae import *
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")
print('Using device', device)



model_name = 'vae_V3-M2T-kl0.1'
data_root = "../../dataset/VAE_1020_obj4/"
Flag_pre_load_model = True
pretrain_file_name =  "vae_V3-M2T_128"
lr = 1e-4
M2T_FLAG = True

data_num = 1000*12*2 #12*12*1000
test_split = 0.9
save_dir = "."
nepoch = 2000
norm_type = "bn"


# Label Embeddings
num_labels = 12

# Initialize MLP Model
pretrained_model = '../mlp/1117-2/best_1117-2.pth'
mlp_model = MLPModel(input_dim=32 * 4 * 4, label_num=num_labels, hidden_dim=1024, output_dim=32 * 4 * 4,
                     num_res_blocks=2).to(device)
mlp_model.load_state_dict(torch.load(pretrained_model, map_location=device))

block_widths = (1, 4, 8, 16, 32)
batch_size = 128
image_size = 128
ch_multi = 2
num_res_blocks = 1
latent_channels = 32

feature_scale = 1
kl_scale = 0.1


# Custom dataset class that returns two transformed images
class DoubleImageDataset(Dataset):
    def __init__(self, root_dir, data_num, m2t_Flag, transform=None):
        self.input_root_dir = root_dir + 'origin_images_before/'
        self.label_root_dir = root_dir + 'origin_images_after/'
        self.transform = transform
        self.data_num = data_num
        self.m2t_Flag = m2t_Flag

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        if self.m2t_Flag:
            input_embed_id_0 = idx // 144
            input_embed_id_1 = (idx % 144) // 12

            label_embed_id_0 = idx // 144
            label_embed_id_1 = (idx % 144) % 12

            input_img_path = self.input_root_dir+ f'label_{input_embed_id_0}_{input_embed_id_1}.png'
            label_img_path = self.label_root_dir+ f'label_{label_embed_id_0}_{label_embed_id_1}.png'

            input_img = Image.open(input_img_path).convert("RGB")  # Ensure 3-channel images
            target_img = Image.open(label_img_path).convert("RGB")  # Ensure 3-channel images

            if self.transform:
                input_img = self.transform(input_img)
                target_img = self.transform(target_img)

            return input_img, target_img, label_embed_id_1

        else:
            if idx < 12000:
                embed_id_0 = idx//12
                embed_id_1 = idx%12
                img_path = self.input_root_dir+ f'label_{embed_id_0}_{embed_id_1}.png'

            else:
                idx-=12000
                embed_id_0 = idx // 12
                embed_id_1 = idx % 12
                img_path = self.label_root_dir + f'label_{embed_id_0}_{embed_id_1}.png'

            img = Image.open(img_path).convert("RGB")  # Ensure 3-channel images

            if self.transform:
                img = self.transform(img)

            return img, img, idx

# Create dataloaders
# This code assumes there is no pre-defined test/train split and will create one for you
print("-Target Image Size %d" % image_size)
transform = transforms.Compose([transforms.Resize((image_size,image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

def func_kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean() * 0.1


n_train_examples = int(data_num * test_split)
n_test_examples = data_num - n_train_examples

# Wrap the dataset with the custom DoubleImageDataset to get two images per sample
dataset = DoubleImageDataset(root_dir=data_root, data_num=data_num, m2t_Flag=M2T_FLAG, transform=transform)

# Split into training and validation sets (90% train, 10% validation)
train_size = int(test_split * data_num)
val_size = data_num - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
dataiter_train = iter(train_loader)
# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)

test_images, gt_images, test_label_id= next(dataiter)
train_images, train_gt_images, train_label_embed= next(dataiter_train)

# Create AE network.
vae_net = VAE(channel_in=test_images.shape[1],
              ch=ch_multi,
              blocks=block_widths,
              latent_channels=latent_channels,
              num_res_blocks=num_res_blocks,
              norm_type=norm_type,
              M2T_FLAG = M2T_FLAG).to(device)
vae_with_mlp = VAEWithMLP(vae_net, mlp_model).to(device)

# Setup optimizer
optimizer = optim.Adam(list(vae_net.parameters()) + list(mlp_model.parameters()), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

if norm_type == "bn":
    print("-Using BatchNorm")
elif norm_type == "gn":
    print("-Using GroupNorm")
else:
    ValueError("norm_type must be bn or gn")

# Create the feature loss module if required
if feature_scale > 0:
    feature_extractor = VGG19().to(device)
    print("-VGG19 Feature Loss ON")
else:
    feature_extractor = None
    print("-VGG19 Feature Loss OFF")

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in vae_net.parameters():
    num_model_params += param.flatten().shape[0]

print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
fm_size = image_size//(2 ** len(block_widths))
print("-The Latent Space Size Is %dx%dx%d!" % (latent_channels, fm_size, fm_size))

os.makedirs(save_dir + "/Models",exist_ok=True)
os.makedirs(save_dir + "/Results",exist_ok=True)

save_file_name = model_name + "_" + str(image_size)


min_loss= np.inf
patience = 0
threshold_patience = 100

checkpoint = torch.load(save_dir + "/Models/" + pretrain_file_name + ".pt",
                        map_location="cpu")
print("-Checkpoint loaded!")
vae_net.load_state_dict(checkpoint['model_state_dict'])
vae_with_mlp.eval()
with torch.no_grad():
    val_loss = 0
    for i, (images, label_images, label_id) in enumerate(test_loader):
        images,label_images,label_id = images.to(device),label_images.to(device),label_id.to(device)


        recon_img, mu, log_var = vae_with_mlp(images, label_embed_id=label_id)

        loss = F.mse_loss(recon_img, label_images).item()
        val_loss += loss

        img_cat = torch.cat((recon_img.cpu(), label_images.cpu()), 2).float()

        for sample in range()
        save_image(comparison, f"{output_dir}/result_batch_{idx}.png", nrow=4, normalize=True)


        if i == 0:
            vutils.save_image(img_cat,
                              "%s/%s/%s_%d_test.png" % (save_dir,
                                                        "Results",
                                                        model_name,
                                                        image_size),
                              normalize=True)



