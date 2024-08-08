# Implementation of Deep Convolutional GANs from scratch

# DCGAN Training Project

This project demonstrates training a Deep Convolutional Generative Adversarial Network (DCGAN) from scratch. The script allows for training with two different model architectures: a base model and a model with additional layers.

## Table of Contents

- [Project Structure](#project-structure)
  - [GAN files](#gan-files)
  - [Extra Files for GANs](#extra-files-for-gans)
  - [Environment setup and README](#environment-setup-and-readme)
  - [Directories](#directories)
- [Setup](#setup)
  - [Conda Environment](#conda-environment)
  - [Python Virtual Environment](#python-virtual-environment)
- [Usage](#usage)
- [Arguments](#arguments)
- [Logging and Outputs](#logging-and-outputs)

## Project Structure

#### GAN files

- `DCGAN.py`: Entry point to run the DCGAN training with configurable arguments.
- `config.py`: Configuration file to manage default settings and command-line arguments.
- `main.py`: Manages the training process.
- `train.py`: Contains training logic for the models.
- `dataset.py`: Loads and pre-processes the dataset.
- `models.py`: Contains the description of Generator and Discriminator models and also the models having additional layers.

#### Extra Files for GANs

- `utils.py`: Utility functions for saving images, logging, and cleaning directories.
- `all_epochs_generator.py`: This is essentially a collage generator file, which samples every 10 epochs and makes a collage of them.

#### Environment setup and README

- `environment.yml`: This file can be used to setup a - [conda environment](#conda-environment) on your machine having all required libraries for the project in a conda environment (provided you have [anaconda](https://www.anaconda.com/download) or [miniconda](https://docs.anaconda.com/miniconda/) on your device)
- `requirements.txt`: This file can be used to [python virtual environment](#python-virtual-environment) to install the required libraries in a python virtual environment

#### Directories

- `DATASET`: Your Training DATASET can be stored here. Although the code allows you to also put your training data at any other location on your device, and then give its path when giving the command to initiate the training.
- `checkpoint`: This is one of the crucial directories for the project. After the training is done, it will contain one more director inside it - generated_images (if you trained on the base model) or generated_images_more_layers (if you trained on the model with more layers), training_checkpoints, collage_output.png(if you run the all_epochs_generator.py)

# Setup

### Conda Environment

To create a conda environment with the required dependencies, use the provided `environment.yml` file:

```sh
conda env create -f environment.yml
conda activate dcgan-env
```

### Python Virtual Environment

Alternatively, you can use a Python virtual environment and install the dependencies from requirements.txt:

```sh
python -m venv dcgan-env
source dcgan-env/bin/activate # On Windows use `dcgan-env\Scripts\activate`
pip install -r requirements.txt
```

# Usage

## Training the Model

To run the DCGAN training, execute the DCGAN.py script with appropriate arguments:

```sh
python DCGAN.py --model_type 'Type of Model' --epochs 'Number of Epochs' --batch_size 'Batch Size' --noise_dim 'Noise Dimension' --num_examples_to_generate 'Number of Examples to Generate' --DATASET_LOC 'Path of the location where training data is stored' --RESULT_LOC 'Path of Location where you want the results to be stored'
```

### Arguments

It is upto the user whether to give arguments or whether not to. Below is a detailed summary

| Argument                     | Description                                                                                                                                                                                                                    | Default Value if no argument is given |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| `--model_type`               | There are two arguments that you can give here - `base` or `more_layers`. The base layer is the default DCGAN architecture. The more_layers architecture contains one more additional layer in the original DCGAN architecture | `base`                                |
| `Number of Epochs`           | Number of epochs to train the model for can be defined here.                                                                                                                                                                   | `50`                                  |
| `--batch_size`               | Batch size for training can be defined here.                                                                                                                                                                                   | `32`                                  |
| `--noise_dim`                | Dimension of the noise vector to be given to the generator is defined here.                                                                                                                                                    | `100`                                 |
| `--num_examples_to_generate` | Number of examples to generate per epoch can be defined here.                                                                                                                                                                  | `16`                                  |
| `--DATASET_LOC`              | Location where the training dataset is stored.                                                                                                                                                                                 | `./DATASET/IITD_Database/`            |
| `--RESULT_LOC`               | Location where the intermediate results and weights will be stored.                                                                                                                                                            | `./checkpoint/`                       |

<line1>

## Generating Images

# Logging and Outputs

- Generated images during training will be saved by default .`/checkpoint/`generated_images or `./checkpoint/generated_images_more_layers` based on the model type.
- Training logs will be saved in `./checkpoint/training_logs.txt`
- Final training results, including loss plots, will be saved in the same directory.
- On any given time, the folder will contain the results of last time the model was trained and will be cleared everytime a training is started.
