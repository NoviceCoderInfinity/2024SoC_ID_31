from main import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(config['input_dim'], config['hidden_dim'], config['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    for epoch in range(1, config['num_epochs'] + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
