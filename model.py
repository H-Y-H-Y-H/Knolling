import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
    def __init__(self, input_dim=512, label_num=12, hidden_dim=1024, output_dim=(4,32,32), num_res_blocks=2):
        super(MLPModel, self).__init__()
        self.label_embedding = nn.Embedding(label_num, 128)
        self.fc_in = nn.Linear(input_dim + 128, hidden_dim)
        self.act = nn.ReLU()
        self.output_dim = output_dim

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim[0]*output_dim[1]*output_dim[2])

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
        output_latent = output.view(batch_size, self.output_dim[0],self.output_dim[1],self.output_dim[2])
        return output_latent


# MLPModel as a part of the VAE pipeline
class VAEWithMLP(nn.Module):
    def __init__(self, vae, mlp_model: MLPModel):
        super(VAEWithMLP, self).__init__()
        self.vae = vae
        self.mlp_model = mlp_model

    def forward(self, x, label_embed_id=None):
        latents = self.vae.encode(x).latent_dist.sample()


        # Use MLPModel to process the latent space
        processed_latent = self.mlp_model(latents, label_embed_id)

        reconstructed = self.vae.decode(processed_latent).sample

        return reconstructed, processed_latent

