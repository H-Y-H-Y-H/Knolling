import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the LatentDataset class
class LatentDataset(Dataset):
    def __init__(self, data_loaded, data_num):
        self.input_data = data_loaded[0]  # before_data
        self.after_data = data_loaded[1]  # after_data
        self.data_num = data_num

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        input_id = idx // 12
        label_id = 12 * (idx // 144) + idx % 12
        label_embed_id = idx % 12

        input_latent = self.input_data[input_id]
        after_latent = self.after_data[label_id]

        # Convert to torch tensors
        input_latent = torch.from_numpy(input_latent).float()
        after_latent = torch.from_numpy(after_latent).float()
        label_embed_id = torch.tensor(label_embed_id).long()

        return input_latent, after_latent, label_embed_id


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out += identity  # Skip connection
        out = self.act(out)
        return out

# Define the MLPModel class with skip connections
class MLPModel(nn.Module):
    def __init__(self, input_dim=512, label_num=12, hidden_dim=1024, output_dim=512, num_res_blocks=2):
        super(MLPModel, self).__init__()
        self.label_embedding = nn.Embedding(label_num, 128)
        self.fc_in = nn.Linear(input_dim + 128, hidden_dim)
        self.act = nn.ReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_latent, label_embed_id):

        batch_size = input_latent.size(0)
        # Flatten the input latent vector
        input_latent_flat = input_latent.view(batch_size, -1)  # (batch_size, 512)
        # Get label embeddings
        label_emb = self.label_embedding(label_embed_id)       # (batch_size, 128)
        # Concatenate input latent vector and label embedding
        x = torch.cat([input_latent_flat, label_emb], dim=1)   # (batch_size, 640)

        # Input layer
        x = self.act(self.fc_in(x))

        # Residual blocks
        x = self.res_blocks(x)

        # Output layer
        output = self.fc_out(x)

        # Reshape output to match after_latent shape
        output_latent = output.view(batch_size, 32, 4, 4)
        return output_latent


def train_model():

    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    if load_pretrained_flag:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))

    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 100000
    best_val_loss = float('inf')  # Initialize the best validation loss
    train_losses = []
    val_losses = []
    patience_threshold = 200
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_latent, after_latent, label_embed_id in train_loader:
            input_latent = input_latent.to(device)
            after_latent = after_latent.to(device)
            label_embed_id = label_embed_id.to(device)

            # Forward pass
            output_latent = model(input_latent, label_embed_id)

            # Compute loss
            loss = criterion(output_latent, after_latent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_latent.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for input_latent, after_latent, label_embed_id in val_loader:
                input_latent = input_latent.to(device)
                after_latent = after_latent.to(device)
                label_embed_id = label_embed_id.to(device)

                # Forward pass
                output_latent = model(input_latent, label_embed_id)

                # Compute loss
                val_loss = criterion(output_latent, after_latent)

                total_val_loss += val_loss.item() * input_latent.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, Patience Counter: {patience_counter}")

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_folder + f'best_{model_name}.pth')

            # Save the loss values to a file
            np.savetxt(save_folder + 'train_losses.csv', train_losses)
            np.savetxt(save_folder + 'val_losses.csv', val_losses)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience_threshold:
            print("Early stopping due to no improvement in validation loss.")
            break

        scheduler.step(avg_val_loss)

def use_model():

    latent_list = []
    # Validation
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for input_latent, after_latent, label_embed_id in val_loader:
            input_latent = input_latent.to(device)
            after_latent = after_latent.to(device)
            label_embed_id = label_embed_id.to(device)

            # Forward pass
            output_latent = model(input_latent, label_embed_id)
            output_latent = output_latent.detach().cpu()
            latent_list.append(output_latent)

    latent_output = np.concatenate(latent_list)




if __name__ == "__main__":

    # Load data
    before_data = np.load('../dataset/VAE_1020_obj4/before_latent.npy')  # (12000, 32, 4, 4)
    after_data = np.load('../dataset/VAE_1020_obj4/after_latent.npy')  # (12000, 32, 4, 4)
    print('loaded')
    # Create dataset
    data_loaded = [before_data, after_data]
    data_num = 288000  # Total data points 2000*12*12
    dataset = LatentDataset(data_loaded, data_num)
    # Prepare the data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    mode = 0

    # Train model
    if mode == 0:
        model_name = '0211'
        save_folder = f'mlp/{model_name}/'
        os.makedirs(save_folder, exist_ok=True)

        pretrained_model = 'mlp/1117-2/best_1117-2.pth'
        load_pretrained_flag = True

        # Initialize the model, optimizer, and loss function
        model = MLPModel(input_dim=32 * 4 * 4, label_num=12, hidden_dim=1024, output_dim=32 * 4 * 4, num_res_blocks=2).to(device)


        train_model()


    if mode == 1:
        use_model_path = 'mlp/1117/best_1117.pth'
        use_model()



