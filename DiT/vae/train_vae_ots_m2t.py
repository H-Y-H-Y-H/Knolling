from diffusers import AutoencoderKL
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from m2t_model import *
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
model_name = 'vae_V10-M2T'
Flag_pre_load_model = True
pretrain_file_name =  "vae_V10-1"

# Initialize MLP Model
pretrained_model = '../mlp/1117-2/best_1117-2.pth'

num_labels = 12
mlp_model = MLPModel(input_dim=4*32*32, label_num=num_labels, hidden_dim=1024, output_dim=(4,32,32), num_res_blocks=2).to(device)
# mlp_model.load_state_dict(torch.load(pretrained_mlp_model, map_location=device))



data_root = "C:/Users/yuhan/PycharmProjects/knolling_data/dataset/"
batch_size = 128
lr = 1e-4
num_epochs = 1000

num_sample = 2000
each_obj_num_data = num_sample*num_labels
data_num = each_obj_num_data*7
test_split = 0.8
save_dir = "."
image_size =128
threshold_patience=100
patience = 0


# Transform for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])


# Dataset and DataLoader
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



# Load dataset
dataset = DoubleImageDataset(root_dir=data_root, data_num=data_num, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained AutoencoderKL model
# vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
# url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
# vae_model = AutoencoderKL.from_single_file(url).to(device)
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
checkpoint = torch.load(save_dir + "/Models/" + pretrain_file_name + ".pt", map_location="cpu")
vae_model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)

vae_with_mlp = VAEWithMLP(vae_model, mlp_model).to(device)

def print_model_info(model, input_size=(1, 3, 128, 128)):
    # Compute number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {num_params / 1e6:.2f}M parameters")

    # Compute latent size
    with torch.no_grad():
        dummy_input = torch.randn(input_size).to(device)
        latent = model.encode(dummy_input).latent_dist.sample()
        print(f"Latent Size: {latent.shape}")  # Shape of the latent space

# Print model info
print_model_info(vae_with_mlp)


min_loss = np.inf
# Training loop
for epoch in range(num_epochs):
    vae_with_mlp.train()
    train_loss = 0
    for images, label_images, label_id in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        images, label_images, label_id = images.to(device), label_images.to(device), label_id.to(device)

        # Forward pass through VAE
        reconstructed, latents = vae_with_mlp(images, label_id)

        # Reconstruction loss
        loss = torch.nn.functional.mse_loss(reconstructed, label_images)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation
    vae_with_mlp.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (images, label_images,label_id) in enumerate(val_loader):
            images,label_images,label_id = images.to(device),label_images.to(device),label_id.to(device)
            reconstructed, latents = vae_with_mlp(images,label_id)
            loss = torch.nn.functional.mse_loss(reconstructed, label_images)
            val_loss += loss.item()
            if i==0:
                img_cat = torch.cat((reconstructed.cpu(), label_images.cpu()), 2).float()
                vutils.save_image(img_cat,
                                  "%s/%s/%s_%d_test.png" % (save_dir,
                                                            "Results",
                                                            model_name,
                                                            image_size),
                                                            normalize=True)
    test_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {test_loss:.4f}")
    if test_loss < min_loss:
        min_loss = test_loss
        patience = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': vae_with_mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_dir + "/Models/" + model_name + f".pt")

    else:
        patience += 1

        if patience > threshold_patience:
            quit()

    scheduler.step(test_loss)