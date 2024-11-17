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
from vae import VAE
import numpy as np
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")
print('Using device', device)



model_name = 'vae_V3-M2T'
data_root = "../../dataset/VAE_1020_obj4/"
Flag_pre_load_model = True
pretrain_file_name =  "vae_V2_128"
lr = 1e-4
M2T_FLAG = True

data_num = 1000*12*2 #12*12*1000
test_split = 0.9
save_dir = "."
nepoch = 2000
norm_type = "bn"


# Label Embeddings
num_labels = 12
latent_size = (32,4,4)
embedding_dim = latent_size[0] * latent_size[1] * latent_size[2]
label_embedding = nn.Embedding(num_labels, embedding_dim).to(device)
# Check trainability (by default, it's trainable)
print(f"Embeddings trainable: {label_embedding.weight.requires_grad}")
# for param in label_embedding.parameters():
#     param.requires_grad = False



block_widths = (1, 4, 8, 16, 32)
parser = argparse.ArgumentParser(description="Training Params")

# int args
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=128)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=128)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=2)

parser.add_argument("--num_res_blocks", '-nrb',
                    help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)

parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=32)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)

# float args
parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint", default=Flag_pre_load_model)
parser.add_argument("--deep_model", '-dm', action='store_true',
                    help="Deep Model adds an additional res-identity block to each down/up sampling stage")

args = parser.parse_args()


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
print("-Target Image Size %d" % args.image_size)
transform = transforms.Compose([transforms.Resize((args.image_size,args.image_size)),
                                # transforms.CenterCrop(args.image_size),
                                # transforms.RandomHorizontalFlip(0.5),
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
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
dataiter_train = iter(train_loader)
# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)

test_images, gt_images, test_label_id= next(dataiter)
train_images, train_gt_images, train_label_embed= next(dataiter_train)


def imshow(img_tensor, ax, title=None, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    img = img_tensor.clone().detach()  # Clone the tensor to avoid modifying the original one
    img = img.permute(1, 2, 0).numpy()  # Convert from [C, H, W] to [H, W, C]

    # If the image was normalized, denormalize it
    img = img * std + mean  # Reverse normalization assuming the image was normalized to [-1, 1]

    # Clip the image to ensure it's in the valid range [0, 1] for float
    img = img.clip(0, 1)

    ax.imshow(img)
    if title:
        ax.set_title(title)
    ax.axis('off')


# Visualize the test images and ground truth images
fig, axs = plt.subplots(2, 20, figsize=(15, 6))

for i in range(20):
    # Plot test images (row 1)
    imshow(test_images[i], axs[0, i])
    # Plot ground truth images (row 2)
    imshow(gt_images[i], axs[1, i])
    gt_images[i]-=1


plt.tight_layout()
plt.show()

# Create AE network.
vae_net = VAE(channel_in=test_images.shape[1],
              ch=args.ch_multi,
              blocks=block_widths,
              latent_channels=args.latent_channels,
              num_res_blocks=args.num_res_blocks,
              norm_type=norm_type,
              M2T_FLAG = M2T_FLAG).to(device)

# Setup optimizer
optimizer = optim.Adam(list(vae_net.parameters())+list(label_embedding.parameters()), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)

# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

if norm_type == "bn":
    print("-Using BatchNorm")
elif norm_type == "gn":
    print("-Using GroupNorm")
else:
    ValueError("norm_type must be bn or gn")

# Create the feature loss module if required
if args.feature_scale > 0:
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
fm_size = args.image_size//(2 ** len(block_widths))
print("-The Latent Space Size Is %dx%dx%d!" % (args.latent_channels, fm_size, fm_size))

os.makedirs(save_dir + "/Models",exist_ok=True)
os.makedirs(save_dir + "/Results",exist_ok=True)

# Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
# If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
# it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
save_file_name = model_name + "_" + str(args.image_size)

if args.load_checkpoint:
    if os.path.isfile(save_dir + "/Models/" + pretrain_file_name + ".pt"):
        checkpoint = torch.load(save_dir + "/Models/" + pretrain_file_name + ".pt",
                                map_location="cpu")
        print("-Checkpoint loaded!")
        vae_net.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint["epoch"]
        data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
    else:
        raise ValueError("Warning Checkpoint does NOT exist -> check model name or save directory")
else:
        print("Starting from scratch")
        start_epoch = 0
        # Loss and metrics logger
        data_logger = defaultdict(lambda: [])
print("")

min_loss= np.inf
patience = 0
threshold_patience=100

# Start training loop
for epoch in range(nepoch):
    print('Epoch: ', epoch)
    vae_net.train()
    for i, (images, label_images, label_id) in enumerate(train_loader):
        images,label_images,label_id = images.to(device),label_images.to(device),label_id.to(device)

        bs, c, h, w = images.shape

        # We will train with mixed precision!
        with torch.cuda.amp.autocast():
            if M2T_FLAG:
                label_embed_tensor = label_embedding(label_id)
                # Reshape the embeddings to match the latent vector size: 32x4x4
                label_embed_tensor = label_embed_tensor.view(-1, *latent_size)
                recon_img, mu, log_var = vae_net(images,label_latent = label_embed_tensor)
            else:
                recon_img, mu, log_var = vae_net(images)


            kl_loss = func_kl_loss(mu, log_var)
            mse_loss = F.mse_loss(recon_img, label_images)
            loss = args.kl_scale * kl_loss + mse_loss

            # Perception loss
            if feature_extractor is not None:
                feat_in = torch.cat((recon_img, label_images), 0)
                feature_loss = feature_extractor(feat_in)
                loss += args.feature_scale * feature_loss
                data_logger["feature_loss"].append(feature_loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()

        # Log losses and other metrics for evaluation!
        data_logger["mu"].append(mu.mean().item())
        data_logger["mu_var"].append(mu.var().item())
        data_logger["log_var"].append(log_var.mean().item())
        data_logger["log_var_var"].append(log_var.var().item())

        data_logger["kl_loss"].append(kl_loss.item())
        data_logger["img_mse"].append(mse_loss.item())

        # Save results and a checkpoint at regular intervals

    print("---Train---", "kl_loss:", kl_loss.item(), "img_mse:", mse_loss.item())



    # In eval mode the model will use mu as the encoding instead of sampling from the distribution
    vae_net.eval()
    for i, (images, label_images, label_id) in enumerate(test_loader):
        images,label_images,label_id = images.to(device),label_images.to(device),label_id.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Save an example from testing and log a test loss
                if M2T_FLAG:
                    label_embed_tensor = label_embedding(label_id)
                    # Reshape the embeddings to match the latent vector size: 32x4x4
                    label_embed_tensor = label_embed_tensor.view(-1, *latent_size)
                    recon_img, mu, log_var = vae_net(images, label_latent=label_embed_tensor)
                else:
                    recon_img, mu, log_var = vae_net(images)
                loss = F.mse_loss(recon_img, label_images.to(device)).item()

                data_logger['test_mse_loss'].append(loss)
                if i==0:
                    img_cat = torch.cat((recon_img.cpu(), label_images.cpu()), 2).float()
                    vutils.save_image(img_cat,
                                      "%s/%s/%s_%d_test.png" % (save_dir,
                                                                "Results",
                                                                model_name,
                                                                args.image_size),
                                                                normalize=True)

    test_loss = np.mean(data_logger['test_mse_loss'])
    print("---Test---", "img_mse: ", test_loss, "patience: ",patience)
    if test_loss < min_loss:
        min_loss = test_loss
        patience = 0
        torch.save({
            'epoch': epoch + 1,
            'data_logger': dict(data_logger),
            'model_state_dict': vae_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_dir + "/Models/" + save_file_name + f".pt")

    else:
        patience += 1

        if patience > threshold_patience:
            quit()

    scheduler.step(loss)