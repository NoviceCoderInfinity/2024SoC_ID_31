import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderDecoderDecoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)  # Output is for both mu and logvar
        )
        
        # First Decoder (after encoder)
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Second Decoder (after first decoder)
        self.decoder2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding stage
        mu_logvar = self.encoder(x)
        batch_size = mu_logvar.size(0)
        latent_dim = mu_logvar.size(1) // 2
        mu_logvar = mu_logvar.view(batch_size, 2, latent_dim)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mu

        # First decoding stage
        intermediate_x = self.decoder1(z)

        # Second decoding stage
        final_x = self.decoder2(intermediate_x)
        
        return final_x, mu, logvar