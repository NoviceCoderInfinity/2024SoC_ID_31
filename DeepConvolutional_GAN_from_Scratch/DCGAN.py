import subprocess
import argparse

data_dir = './DATASET/celebA_HQ/'
result_dir = './checkpoint/'
image_size = (64, 64)

# Default values
batch_size = 32
epochs = 50
noise_dim = 100
num_examples_to_generate = 16
model_type = 'base'

def main(data_dir, result_dir, batch_size, epochs, noise_dim, num_examples_to_generate, model_type):
    parser = argparse.ArgumentParser(description="Run DCGAN training.")
    parser.add_argument('--model_type', type=str, choices=['base', 'more_layers'], help='Model type to use')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--noise_dim', type=int, help='Dimension of the noise vector')
    parser.add_argument('--num_examples_to_generate', type=int, help='Number of examples to generate')
    parser.add_argument('--DATASET_LOC', type=str, help='Location where the Training Dataset is stored')
    parser.add_argument('--RESULT_LOC', type=str, help='Location where the Intermediate Results, weights of the training data will be stored')

    args = parser.parse_args()
    model_type = args.model_type if args.model_type else model_type
    epochs = args.epochs if args.epochs else epochs
    batch_size = args.batch_size if args.batch_size else batch_size
    noise_dim = args.noise_dim if args.noise_dim else noise_dim
    num_examples_to_generate = args.num_examples_to_generate if args.num_examples_to_generate else num_examples_to_generate
    data_dir = args.DATASET_LOC if args.DATASET_LOC else data_dir
    result_dir = args.RESULT_LOC if args.RESULT_LOC else result_dir

    return subprocess.run(['python3', 'main.py', '--model_type', model_type, '--epochs', str(epochs), '--batch_size', str(batch_size), '--noise_dim', str(noise_dim), '--num_examples_to_generate', str(num_examples_to_generate), '--DATASET_LOC', data_dir, '--RESULT_LOC', result_dir])

if __name__ == "__main__":
    main(data_dir, result_dir, batch_size, epochs, noise_dim, num_examples_to_generate, model_type)
