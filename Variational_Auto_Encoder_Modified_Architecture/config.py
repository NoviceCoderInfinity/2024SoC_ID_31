import argparse

def get_config():
    parser = argparse.ArgumentParser(description='VAE')

    # Data parameters
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')

    # Model parameters
    parser.add_argument('--input_dim', type=int, default=784, help='dimensionality of input data')
    parser.add_argument('--hidden_dim', type=int, default=400, help='dimensionality of hidden layer')
    parser.add_argument('--latent_dim', type=int, default=20, help='dimensionality of latent space')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training')

    config = parser.parse_args()
    return config
