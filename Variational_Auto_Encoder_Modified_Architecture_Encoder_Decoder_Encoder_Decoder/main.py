import torch
from config import get_config
from dataset import get_dataloader
from models import VAE
from train import train
from utils import set_seed, visualize_reconstruction, load_model, save_model, plot_images

def main():
    config = get_config()
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloader(config.batch_size, config.num_workers)
    model = VAE(config.input_dim, config.hidden_dim, config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Train the model and save weights
    train(model, train_loader, optimizer, config.num_epochs, device)
    
    # Visualize reconstruction
    visualize_reconstruction(model, test_loader, device)

    # Save model weights
    save_path = 'vae.pth'
    save_model(model, save_path)
    print(f"Model saved to {save_path}")

    # Load model weights
    load_model(model, save_path, device)
    print("Model loaded from", save_path)

    # Generate new images
    with torch.no_grad():
        z = torch.randn(16, config.latent_dim).to(device)
        new_images = model.decoder2(z).cpu().numpy()

    plot_images(new_images, title="Generated Images")

if __name__ == "__main__":
    main()
