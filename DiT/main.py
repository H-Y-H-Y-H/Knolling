import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchsummary import summary  # For model size and parameters
import numpy as np
# Use your modified DiT class and helpers from the original code
from models import *

class LatentDataset(Dataset):
    def __init__(self, data_loaded, data_num):
        self.input_data = data_loaded[0]
        self.after_data = data_loaded[1]

        self.data_num = data_num

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        input_id = idx // 12
        lable_id = 12*(idx//144) + idx % 12
        label_embed_id = idx % 12

        input_latent = self.input_data[input_id]
        after_latent = self.after_data[lable_id]

        return input_latent, after_latent, label_embed_id



def validate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Timesteps and dummy class labels
            timesteps = torch.randint(0, 1000, (inputs.size(0),), device=device)
            labels = torch.randint(0, 1000, (inputs.size(0),), device=device)

            # Forward pass
            outputs = model(inputs, timesteps, labels)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=5):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0

        for inputs, targets, embed_id in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets, embed_id = inputs.to(device), targets.to(device), embed_id.to(device)
            optimizer.zero_grad()

            # Timesteps and dummy class labels
            timesteps = torch.randint(0, N_time_steps, (inputs.size(0),), device=device)

            # Forward pass
            outputs = model(inputs, timesteps, embed_id)

            # Calculate loss
            loss = criterion(outputs, targets)
            loss.backward()

            # Optimize
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validate the model
        val_loss = validate_model(model, val_loader, criterion, device)

        # Save the model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model")

if __name__ == "__main__":
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device',device)

    # Hyperparameters
    batch_size = 128
    img_size = 128
    epochs = 5
    learning_rate = 1e-4
    N_time_steps = 1000
    data_num = 12000*12

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    before_data = np.load('../../dataset/VAE_1020_obj4/before_latent.npy')
    after_data = np.load('../../dataset/VAE_1020_obj4/after_latent.npy')

    # Load the dataset from your folder
    dataset = LatentDataset(data_loaded=[before_data, after_data], data_num=data_num)

    # Split into training and validation sets (90% train, 10% validation)
    train_size = int(0.8 * data_num)
    val_size = data_num - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, criterion, and optimizer setup
    model = DiT_S_2(input_size=img_size // 8, num_classes=1000).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs)
