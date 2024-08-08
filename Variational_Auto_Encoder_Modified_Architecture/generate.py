import torch
from models import VAE
from utils import load_model, plot_images
from config import get_config

def generate_images(model, latent_dim, num_images, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        new_images = model.decoder(z).cpu().numpy()
    plot_images(new_images, title="Generated Images")

if __name__ == "__main__":
    config = get_config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    model = VAE(config.input_dim, config.hidden_dim, config.latent_dim).to(device)
    
    load_model(model, 'vae.pth', device)
    generate_images(model, config.latent_dim, num_images=16, device=device)
