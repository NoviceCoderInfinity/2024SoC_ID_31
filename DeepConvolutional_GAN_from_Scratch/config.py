# config.py
# This file contains the configuration for the DCGAN training.
import argparse

def read_args():
    parser = argparse.ArgumentParser(description="Configuration for DCGAN training.")
    parser.add_argument('--model_type', type=str, choices=['base', 'more_layers'], help='Model type to use', default='base')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=50)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=32)
    parser.add_argument('--noise_dim', type=int, help='Dimension of the noise vector', default=100)
    parser.add_argument('--num_examples_to_generate', type=int, help='Number of examples to generate', default=16)
    parser.add_argument('--DATASET_LOC', type=str, help='Location where the Training Dataset is stored', default='./DATASET/IITD_Database/')
    parser.add_argument('--RESULT_LOC', type=str, help='Location where the Intermediate Results, weights of the training data will be stored', default='./checkpoint/')
    parser.add_argument('--image_size', type=tuple, help='Size of the images', default=(64, 64))

    args = parser.parse_args()
    return args

args = read_args()

model_type = args.model_type
epochs = args.epochs
batch_size = args.batch_size
noise_dim = args.noise_dim
num_examples_to_generate = args.num_examples_to_generate
data_dir = args.DATASET_LOC
result_dir = args.RESULT_LOC
image_size = args.image_size
