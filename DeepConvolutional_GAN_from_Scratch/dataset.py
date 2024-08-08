import os
from glob import glob
import tensorflow as tf
from config import data_dir, batch_size, image_size

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)  # Decode any image format
    image = tf.image.resize(image, image_size)
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

def load_dataset():
    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for extension in valid_extensions:
            image_paths.extend(glob(os.path.join(root, extension)))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory {data_dir} or its subdirectories.")
    
    print(f"Found {len(image_paths)} images.")  # Log the number of found images
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    buffer_size = max(len(image_paths), 1)  # Ensure buffer_size is at least 1
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset
