import torch

from m2t_model import *
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = 'mlp-1'
data_root = "C:/Users/yuhan/PycharmProjects/knolling_data/dataset_white/"

num_labels = 12
num_sample = 2000
each_obj_num_data = num_sample*num_labels*12
data_num = each_obj_num_data*8


test_split = 0.8
save_dir = "."
image_size =128
threshold_patience=100
patience = 0
batch_size = 512
lr = 1e-4
num_epochs = 1000


class LatentDataset(Dataset):
    def __init__(self, root_dir, data_num):
        self.root_dir = root_dir
        self.data_num = data_num

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        obj_num_id = idx//each_obj_num_data
        self.root_data_path = self.root_dir + f'obj{obj_num_id + 2}/'
        idx = idx - obj_num_id * each_obj_num_data

        input_embed_id_0 = idx // (num_labels**2)
        input_embed_id_1 = (idx % (num_labels**2)) // num_labels

        label_embed_id_0 = idx // (num_labels**2)
        label_embed_id_1 = (idx %  (num_labels**2)) % num_labels

        input_latent_path = self.root_data_path + f'/latent_before/label_{input_embed_id_0}_{input_embed_id_1}.npy'
        label_latent_path = self.root_data_path + f'/latent_after/label_{label_embed_id_0}_{label_embed_id_1}.npy'

        input_latent = np.load(input_latent_path)
        target_latent = np.load(label_latent_path)
        input_latent = torch.Tensor(input_latent)
        target_latent = torch.Tensor(target_latent)

        return input_latent, target_latent, label_embed_id_1


# Transform for images
transform = transforms.Compose([
    transforms.ToTensor(),
])


torch.manual_seed(0)
# Load dataset
dataset = LatentDataset(root_dir=data_root, data_num=data_num)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

mlp_model = MLPModel(input_dim=4*32*32, label_num=num_labels, hidden_dim=1024, output_dim=(4,32,32), num_res_blocks=2).to(device)
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)


min_loss = np.inf
# Training loop
for epoch in range(num_epochs):
    mlp_model.train()
    train_loss = 0
    for latent, label_latent, label_id in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        latent, label_latent, label_id = latent.to(device), label_latent.to(device), label_id.to(device)

        # Forward pass through VAE
        prediction = mlp_model(latent, label_id)

        # Reconstruction loss
        loss = torch.nn.functional.mse_loss(prediction, label_latent)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation
    mlp_model.eval()
    val_loss = 0
    with torch.no_grad():
        for latent, label_latent, label_id in tqdm(val_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            latent, label_latent, label_id = latent.to(device),label_latent.to(device),label_id.to(device)
            prediction = mlp_model(latent,label_id)
            loss = torch.nn.functional.mse_loss(prediction, label_latent)
            val_loss += loss.item()
        test_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {test_loss:.4f}")
    if test_loss < min_loss:
        min_loss = test_loss
        patience = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': mlp_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_dir + "/Models/" + model_name + f".pt")

    else:
        patience += 1

        if patience > threshold_patience:
            quit()

    scheduler.step(test_loss)


