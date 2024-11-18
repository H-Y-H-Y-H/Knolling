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
        self.data_num = data_num # 12*12*1000

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

# Define the SinusoidalPositionEmbeddings class
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        embeddings = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=device) / (half_dim - 1)
        )
        embeddings = timesteps.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

# Define the DiffusionModel class
class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=1024, label_num=12):
        super(DiffusionModel, self).__init__()
        self.latent_dim = latent_dim

        self.label_embedding = nn.Embedding(label_num, latent_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        # Input layers
        self.conv1 = nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1)

        self.act = nn.ReLU()

        # Skip layers
        self.skip1 = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)  # Match dimensions to hidden_dim
        self.skip2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)  # Match dimensions to hidden_dim
        self.skip3 = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)  # Match dimensions to latent_dim

    def forward(self, x_t, t, input_latent, label_embed_id):
        batch_size = x_t.shape[0]

        # Time embedding
        t_emb = self.time_embedding(t)  # [batch_size, latent_dim]
        t_emb = t_emb.view(batch_size, self.latent_dim, 1, 1)  # [batch_size, latent_dim, 1, 1]

        # Label embedding
        label_emb = self.label_embedding(label_embed_id)  # [batch_size, latent_dim]
        label_emb = label_emb.view(batch_size, self.latent_dim, 1, 1)  # [batch_size, latent_dim, 1, 1]

        # Combine embeddings
        h = x_t + input_latent + t_emb + label_emb  # Broadcasting addition

        # First layer with skip connection
        h1 = self.act(self.conv1(h))
        skip1_out = self.skip1(h)  # Skip connection
        h1 = h1 + skip1_out  # Add skip connection output

        # Second layer with skip connection
        h2 = self.act(self.conv2(h1))
        skip2_out = self.skip2(h1)  # Skip connection
        h2 = h2 + skip2_out  # Add skip connection output

        # Final layer with skip connection
        h3 = self.conv3(h2)
        skip3_out = self.skip3(h2)  # Skip connection
        h3 = h3 + skip3_out  # Add skip connection output

        return h3  # Output is the predicted noise


# Training function
def train_model():



    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
    criterion = nn.MSELoss()



    # Training loop
    num_epochs = 100000
    best_val_loss = float('inf')  # Initialize the best validation loss
    train_losses = []
    val_losses = []
    patience_threshold = 300
    pati = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_latent, after_latent, label_embed_id in train_loader:
            input_latent = input_latent.to(device)
            after_latent = after_latent.to(device)
            label_embed_id = label_embed_id.to(device)

            batch_size = input_latent.size(0)
            t = torch.randint(0, T, (batch_size,), device=device).long()

            # Sample noise
            noise = torch.randn_like(after_latent)

            # Compute x_t
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
            x_t = sqrt_alphas_cumprod_t * after_latent + sqrt_one_minus_alphas_cumprod_t * noise

            # Predict noise
            noise_pred = model(x_t, t, input_latent, label_embed_id)

            # Compute loss
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

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

                batch_size = input_latent.size(0)
                t = torch.randint(0, T, (batch_size,), device=device).long()

                # Sample noise
                noise = torch.randn_like(after_latent)

                # Compute x_t
                sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
                x_t = sqrt_alphas_cumprod_t * after_latent + sqrt_one_minus_alphas_cumprod_t * noise

                # Predict noise
                noise_pred = model(x_t, t, input_latent, label_embed_id)

                # Compute loss
                val_loss = criterion(noise_pred, noise)

                total_val_loss += val_loss.item() * batch_size

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Patience: {pati}")

        # Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_folder+f'best_{model_name}.pth')

            # Save the loss values to a file
            np.savetxt(save_folder+'train_losses.csv',train_losses)
            np.savetxt(save_folder+'test_losses.csv',val_losses)
            pati = 0

        else:
            pati +=1
        if pati > patience_threshold:
            break
        scheduler.step(avg_val_loss)



def use_model(input_latent,timesteps=1000):
    """
        Deploys the trained model for inference using the iterative denoising process.

        Args:
            input_latent (numpy.ndarray): Input latent vector (before state).

        Returns:
            numpy.ndarray: The generated denoised output.
        """
    label_embed_id = 0 # (0-11)

    model.eval()

    input_latent = torch.tensor(input_latent).float().to(device).unsqueeze(0)  # Add batch dimension
    label_embed_id = torch.tensor(label_embed_id).long().to(device).unsqueeze(0)  # Add batch dimension

    # Start from pure noise
    x_t = torch.randn_like(input_latent).to(device)

    # Iteratively denoise
    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([t], device=device).long()  # Current timestep
        sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)

        with torch.no_grad():
            noise_pred = model(x_t, t_tensor, input_latent, label_embed_id)

        # Reverse diffusion step
        if t > 0:
            beta_t = betas[t]
            noise_scale = torch.sqrt(beta_t)
            x_t = (x_t - noise_pred * noise_scale) / torch.sqrt(1 - beta_t)
        else:
            x_t = (x_t - noise_pred) / sqrt_alpha_t

    # Return the denoised result as a numpy array
    return x_t.squeeze(0).cpu().numpy()





if __name__ == "__main__":


    model_name = '1119'

    save_folder = f'{model_name}/'
    os.makedirs(save_folder,exist_ok=True)

    # Load data
    before_data = np.load('../data/14400_latent_data/before_latent.npy')  # (12000, 32, 4, 4)
    after_data = np.load('../data/14400_latent_data/after_latent.npy')    # (12000, 32, 4, 4)
    print('loaded')

    # Create dataset
    data_loaded = [before_data, after_data]
    data_num = 144000  # Total data points
    dataset = LatentDataset(data_loaded, data_num)

    # Prepare the data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize the model, optimizer, and loss function
    latent_dim = 32
    hidden_dim = 1024
    loaded_model = '1119/best_1119.pth'

    model = DiffusionModel(latent_dim=latent_dim, hidden_dim=hidden_dim, label_num=12).to(device)
    # Set up the diffusion parameters
    T = 1000  # Total time steps
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)


    model.load_state_dict(torch.load(loaded_model, map_location=device))

    train_model()


    # Deploy the model
    generated_output = use_model()


