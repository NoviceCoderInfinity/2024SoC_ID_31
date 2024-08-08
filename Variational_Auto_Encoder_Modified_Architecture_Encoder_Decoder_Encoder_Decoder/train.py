import torch
import torch.nn.functional as F

def loss_function(x_hat, x, mu1, logvar1, mu2, logvar2):
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    return BCE + KLD1 + KLD2

from utils import save_model

def train(model, dataloader, optimizer, num_epochs, device, save_path='vae.pth'):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(-1, 28*28).to(device)
            optimizer.zero_grad()
            x_hat, mu1, logvar1, mu2, logvar2 = model(data)
            loss = loss_function(x_hat, data, mu1, logvar1, mu2, logvar2)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
    
    save_model(model, save_path)
    print(f"Model saved to {save_path}")
