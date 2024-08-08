import torch
import torch.nn as nn
import torch.nn.functional as F

class NestedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(NestedVAE, self).__init__()
        
        # First Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)  # mu and logvar
        )
        
        # Second Encoder
        self.encoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)  # mu and logvar
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # First encoding
        mu_logvar1 = self.encoder1(x)
        batch_size = mu_logvar1.size(0)
        latent_dim = mu_logvar1.size(1) // 2
        mu_logvar1 = mu_logvar1.view(batch_size, 2, latent_dim)
        mu1 = mu_logvar1[:, 0, :]
        logvar1 = mu_logvar1[:, 1, :]
        
        # Reparameterization trick after first encoding
        std1 = torch.exp(0.5 * logvar1)
        z1 = torch.randn_like(std1) * std1 + mu1

        # Second encoding
        mu_logvar2 = self.encoder2(z1)
        latent_dim2 = mu_logvar2.size(1) // 2
        mu_logvar2 = mu_logvar2.view(batch_size, 2, latent_dim2)
        mu2 = mu_logvar2[:, 0, :]
        logvar2 = mu_logvar2[:, 1, :]
        
        # Reparameterization trick after second encoding
        std2 = torch.exp(0.5 * logvar2)
        z2 = torch.randn_like(std2) * std2 + mu2

        # Decode the final latent variable
        recon_x = self.decoder(z2)
        return recon_x, mu2, logvar2
