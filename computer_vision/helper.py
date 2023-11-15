import json
import os
from typing import *
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import time
import numpy as np

""" This module contains generic helper functions that
    can easily be reused across the Computer Vision section.
"""

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate optimal grid shape for the samples visualization figure
def visualize_calculate_grid_shape(num_images: int) -> tuple:
    num_rows = math.isqrt(num_images)
    num_cols = math.ceil(num_images / num_rows)
    return (num_rows, num_cols)

# Visualize a number of image samples from a tensorflow dataset
def visualize_classification_image_samples(dataset: tf.data.Dataset, 
                            num_samples: int, 
                            dataset_info: tfds.core.dataset_info.DatasetInfo,
                            g_shape:  Optional[Tuple[int, int]] = None):
    # Create an iterator for the dataset
    iterator = iter(dataset.take(num_samples))
    if g_shape:
        grid_shape = g_shape
    else:
        grid_shape = visualize_calculate_grid_shape(num_images= num_samples)
    class_dict =  {index: name for index, name in enumerate(dataset_info.features['label'].names)}

    # Verify and plot grid
    row_grid, col_grid = grid_shape

    if row_grid * col_grid == num_samples and num_samples > 7 and row_grid !=1 and col_grid !=1:
        # Create a subplot for displaying images in the grid
        fig, axes = plt.subplots(*grid_shape, figsize=(7, 7))

        for i in range(num_samples):
            image, label = next(iterator)  
            image = (image).numpy().astype(int)  
            label = (label).numpy().astype(int)

            # Define the position of the current subplot in the grid
            row, col = divmod(i, grid_shape[1])

            # Display the image in the current subplot
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Sample: {i}', fontsize=8, color='white')
            axes[row, col].axis('off')
            axes[row, col].set_xticks([])  
            axes[row, col].set_yticks([])  
            axes[row, col].text(0.5,+40, f'Class: {class_dict[label]}', fontsize=8, color='white')
            axes[row, col].set_aspect('equal')
        
        fig.suptitle(f"{dataset_info.name} dataset samples", color='white', y = 0.98)
        fig.tight_layout()
        fig.patch.set_alpha(0)

        plt.show()
    else:
        print(f"Can't calculate grid shape..Please provide a shape or different number of samples..")

# Visualize a number of predictions for a model
def visualize_predictions(images: np.ndarray, 
                          true_labels: list, 
                          predictions: np.ndarray, 
                          dataset_info: tfds.core.dataset_info.DatasetInfo, 
                          num_samples: int,
                          g_shape:  Optional[Tuple[int, int]] = None):
    class_names = dataset_info.features['label'].names
    if g_shape:
        grid_shape = g_shape
    else:
        grid_shape = visualize_calculate_grid_shape(num_images= num_samples)

    # Verify and plot grid
    row_grid, col_grid = grid_shape

    if row_grid * col_grid == num_samples and num_samples > 7 and row_grid !=1 and col_grid !=1:
        fig, axes = plt.subplots(*grid_shape, figsize=(7,7))

        for i in range(num_samples):
            true_label = true_labels[i]
            predicted_label = np.argmax(predictions[i])

            # Define the position of the current subplot in the grid
            row, col = divmod(i, grid_shape[1])

            # Display the image in the current subplot
            axes[row, col].imshow(images[i], cmap='gray')  
            axes[row, col].axis('off')
            axes[row, col].set_xticks([])  
            axes[row, col].set_yticks([])  
            color = 'green' if true_label == predicted_label else 'red'
            axes[row, col].set_title(f'True: {class_names[true_label]}\nPredicted: {class_names[predicted_label]}',
                                      color=color,
                                      fontsize=10)
            axes[row, col].set_aspect('equal')
        
        fig.suptitle(f"{dataset_info.name} model predictions", color='white', y = 0.98)
        fig.tight_layout()
        fig.patch.set_alpha(0)

        plt.subplots_adjust(wspace=0.5, hspace=0)
        plt.show()
    else:
         print(f"Can't calculate grid shape..Please provide a shape or different number of samples..")

# Fast benchmark for input pipelines function
def fast_benchmark(dataset: tf.data.Dataset, num_epochs: int =2):
    start_time = time.perf_counter()
    for _ in tf.data.Dataset.range(num_epochs):
        for _ in dataset:
            pass
    tf.print("Execution time:", time.perf_counter() - start_time)

# Initialize or set the parameters of a model's config dict
def set_model_config(model_name: str, batch_size: int =None, training_epochs: int=None,
                     n_classes: int =None, learning_rate: float = None, optimizer: str = None, val_size: float = None):
    """ Helper function that initializes configs for various model's from 
        templates or changes some parameter in a config.

        Parameters:
        model_name : The name of the model for which to retrieve or modify configurations.
        batch_size : The batch size for training. If provided, updates the batch size in the configuration.
        training_epochs : The number of training epochs. If provided, updates the training epochs in the configuration.
        n_classes : The number of classes in the model. If provided, updates the number of classes in the configuration.
        learning_rate : The learning rate for training. If provided, updates the learning rate in the configuration.
        optimizer : The optimizer used for training. If provided, updates the optimizer in the configuration.
        val_size : The validation set size. If provided, updates the validation set size in the configuration.

    Returns:
        dict: A dictionary containing the updated or initialized model configuration.
    """
    # Default configs
    config_template = {
        "cifar_10": {
            "batch_size": 128,
            "training_epochs": 5,
            "n_classes": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "val_size": 0.2
        },
        # Add configurations for other models as needed
    }

    # Check if the provided model_name is in the template
    if model_name in config_template:
        model_config = config_template[model_name]

        # Update the configuration with new values if provided
        model_config['batch_size'] = batch_size if batch_size is not None else model_config['batch_size']
        model_config['training_epochs'] = training_epochs if training_epochs is not None else model_config['training_epochs']
        model_config['n_classes'] = n_classes if n_classes is not None else model_config['n_classes']
        model_config['learning_rate'] = learning_rate if learning_rate is not None else model_config['learning_rate']
        model_config['optimizer'] = optimizer if optimizer is not None else model_config['optimizer']
        model_config['val_size'] = val_size if val_size is not None else model_config['val_size']

        return model_config
    else:
        raise ValueError(f"Configuration not found for model: {model_name}")
    

# Plot training loss function
def plot_loss(history):
    """
    Plot the training and validation loss over epochs.

    Parameters:
        history (tensorflow.python.keras.callbacks.History): The training history obtained from model.fit.

    Returns:
        None
    """
    # Get the training and validation loss from the history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the training and validation loss with lines
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')  
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss') 
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    max_ticks = 5
    tick_positions = list(range(1, len(train_loss) + 1, max(len(train_loss) // max_ticks, 1)))
    plt.xticks(tick_positions)

    plt.show()