# utils.py
# Description: This file contains utility functions for the project
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from config import image_size, num_examples_to_generate, noise_dim, batch_size, epochs, result_dir, model_type

def generate_and_save_images(model, epoch, test_input, result_dir):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i] + 1) / 2)
        plt.axis('off')

    plt.savefig(os.path.join(result_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close(fig)

def save_model(model, result_dir, epoch):
    save_path = os.path.join(result_dir, f'model_epoch_{epoch}.h5')
    model.save(save_path)
    print(f'Model saved to {save_path}')

def save_logs(logs, file_path):
    df = pd.DataFrame(logs)
    df.to_csv(file_path, index=False)

def plot_logs(logs, save_path):
    df = pd.DataFrame(logs)
    plt.figure(figsize=(10, 5))
    plt.plot(df['batch'], df['gen_loss'], label='Generator Loss')
    plt.plot(df['batch'], df['disc_loss'], label='Discriminator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()



def clear_image_dir(image_dir):
    if not os.path.exists(image_dir):
        return
    for file in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def clear_log_dir(log_dir):
    if not os.path.exists(log_dir):
        return
    for file in os.listdir(log_dir):
        if file.startswith('training_logs') or file.startswith('training_plot'):
            file_path = os.path.join(log_dir, file)
            os.remove(file_path)

def clear_model_files(log_dir):
    if not os.path.exists(log_dir):
        return
    for file in os.listdir(log_dir):
        if file.startswith('generator_model') or file.startswith('discriminator_model'):
            file_path = os.path.join(log_dir, file)
            os.remove(file_path)

def create_directories(image_dir, log_dir):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

def create_log_file(log_file_path, model_type, batch_size, epochs, noise_dim, num_examples_to_generate):
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Training started at: {time.ctime()}\n")
        log_file.write(f"Model: {model_type}\n")
        log_file.write(f"Batch size: {batch_size}\n")
        log_file.write(f"Epochs: {epochs}\n")
        log_file.write(f"Noise dimension: {noise_dim}\n")
        log_file.write(f"Number of examples to generate: {num_examples_to_generate}\n")

def append_end_time(log_file_path):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Training ended at: {time.ctime()}\n")