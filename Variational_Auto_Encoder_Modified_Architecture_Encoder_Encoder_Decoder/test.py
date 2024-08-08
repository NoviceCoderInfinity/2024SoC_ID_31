import torch
import torch.nn.functional as F
from dataset import get_dataloader
from models import NestedVAE
from utils import load_model
from config import get_config

def test(model, dataloader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            data = data.view(-1, 28*28).to(device)
            x_hat, mu, logvar = model(data)
            loss = F.binary_cross_entropy(x_hat, data, reduction='sum') + \
                   -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            test_loss += loss.item()
    
    test_loss /= len(dataloader.dataset)
    print(f'Test Loss: {test_loss}')
    return test_loss

if __name__ == "__main__":
    config = get_config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    _, test_loader = get_dataloader(config.batch_size, config.num_workers)
    model = NestedVAE(config.input_dim, config.hidden_dim, config.latent_dim).to(device)
    
    load_model(model, 'vae.pth', device)
    test(model, test_loader, device)
