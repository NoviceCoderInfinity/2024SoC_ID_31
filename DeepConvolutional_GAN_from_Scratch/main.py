import time
import os
from train import train, train_with_more_layers
from dataset import load_dataset
from config import model_type, batch_size, epochs, noise_dim, num_examples_to_generate, result_dir
from utils import clear_image_dir, clear_log_dir, clear_model_files, create_directories, create_log_file, append_end_time

if __name__ == "__main__":
    # Directory to save generated images and logs
    if model_type == 'more_layers':
        image_dir = os.path.join(result_dir, 'generated_images_more_layers')
    else:
        image_dir = os.path.join(result_dir, 'generated_images')

    log_dir = result_dir

    # Clear directories
    clear_image_dir(image_dir)
    clear_log_dir(log_dir)
    clear_model_files(log_dir)

    # Create directories and log files
    create_directories(image_dir, log_dir)
    log_file_path = os.path.join(log_dir, 'training_logs.txt')
    create_log_file(log_file_path, model_type, batch_size, epochs, noise_dim, num_examples_to_generate)

    # Load dataset
    dataset = load_dataset()

    print(f"\n\n========================================================\n\n"
          f"Training started at: {time.ctime()}"
          f"\nModel: {model_type}"
          f"\nBatch size: {batch_size}"
          f"\nEpochs: {epochs}"
          f"\nNoise dimension: {noise_dim}"
          f"\nNumber of examples to generate: {num_examples_to_generate}\n\n"
          f"\n\n========================================================\n\n")

    # Train the selected model
    if model_type == 'base':
        train(dataset, epochs, log_dir, image_dir, batch_size, noise_dim, num_examples_to_generate)
    elif model_type == 'more_layers':
        train_with_more_layers(dataset, epochs, log_dir, image_dir, batch_size, noise_dim, num_examples_to_generate)

    # Append end time to log file
    append_end_time(log_file_path)
