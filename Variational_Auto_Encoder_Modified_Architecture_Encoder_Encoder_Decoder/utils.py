import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_model(model, filepath):
    """Save the model weights to a file."""
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath, device):
    """Load the model weights from a file."""
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)

def plot_images(images, title, n_row=2, n_col=8):
    """Plot a grid of images."""
    fig, axes = plt.subplots(n_row, n_col, figsize=(2 * n_col, 2 * n_row))
    for i in range(n_row):
        for j in range(n_col):
            axes[i, j].imshow(images[i * n_col + j].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')
    fig.suptitle(title)
    plt.show()

def visualize_reconstruction(model, dataloader, device, num_images=16):
    """Visualize the original and reconstructed images."""
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.view(-1, 28*28).to(device)
    
    with torch.no_grad():
        recon_images, _, _ = model(images)

    images = images.cpu().numpy()[:num_images]
    recon_images = recon_images.cpu().numpy()[:num_images]

    plot_images(images, title="Original Images")
    plot_images(recon_images, title="Reconstructed Images")
