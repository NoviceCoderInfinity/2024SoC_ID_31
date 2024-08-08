import tensorflow as tf
import time
import os
from dataset import load_dataset
from models import build_generator, build_discriminator, build_generator_with_more_layers, build_discriminator_with_more_layers
from utils import generate_and_save_images, save_model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, noise_dim, batch_size):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs, log_dir, image_dir, batch_size, noise_dim, num_examples_to_generate):
    generator = build_generator(noise_dim)
    discriminator = build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    checkpoint_dir = os.path.join(log_dir, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, noise_dim, batch_size)

        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch + 1, seed, image_dir)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec | Gen Loss: {gen_loss} | Disc Loss: {disc_loss}')

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed, image_dir)
    save_model(generator, discriminator, log_dir)

def train_with_more_layers(dataset, epochs, log_dir, image_dir, batch_size, noise_dim, num_examples_to_generate):
    generator = build_generator_with_more_layers(noise_dim)
    discriminator = build_discriminator_with_more_layers()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    checkpoint_dir = os.path.join(log_dir, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, noise_dim, batch_size)

        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch + 1, seed, image_dir)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec | Gen Loss: {gen_loss} | Disc Loss: {disc_loss}')

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed, image_dir)
    save_model(generator, discriminator, log_dir)
