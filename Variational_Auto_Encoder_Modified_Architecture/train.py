import torch
import torch.nn.functional as F

def loss_function(x_hat, x, mu, logvar):
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

from utils import save_model

def train(model, dataloader, optimizer, num_epochs, device, save_path='vae.pth'):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(-1, 28*28).to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(data)
            loss = loss_function(x_hat, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
    
    save_model(model, save_path)
    print(f"Model saved to {save_path}")
