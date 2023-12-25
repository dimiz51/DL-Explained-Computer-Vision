import json
import os
from typing import *
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras_cv import bounding_box
from keras_cv import visualization
from sklearn.utils.class_weight import compute_class_weight

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

# Visualize the Keras-CV compatible dataset
def visualize_object_detection_samples(inputs, value_range, rows, cols, bounding_box_format, class_mapping):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )
    
# Visualize a number of image samples from a tensorflow segmentation dataset
def visualize_segmentation_image_samples(dataset, num_samples=3):
    sample_dataset = dataset.take(num_samples)

    # Create a subplot for displaying images in the grid
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

    # Iterate through the samples dataset
    for i, batch in enumerate(sample_dataset):
        image = batch['image'].numpy().astype(int)
        label = batch['label'].numpy().astype(int)
        segmentation_mask = batch['segmentation_mask'].numpy().astype(int)

        # Plot original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original Image - Sample {i + 1}')
        axes[i, 0].axis('off')

        # Plot segmentation mask
        axes[i, 1].imshow(segmentation_mask[:, :, 0], cmap='gray')
        axes[i, 1].set_title(f'Segmentation Mask - Sample {i + 1}')
        axes[i, 1].axis('off')

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
    else:
        print(f"Can't calculate grid shape..Please provide a shape or different number of samples..")

# Visualize a number of predictions of a classification model
def visualize_classification_predictions(images: np.ndarray, 
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
        plt.subplots_adjust(wspace=0.5, hspace=0)
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
                     n_classes: int =None, learning_rate: float = None, optimizer: str = None, 
                     val_size: float = None, global_clipnorm: int = None):
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
            "training_epochs": 30,
            "n_classes": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "val_size": 0.2,
            'global_clipnorm': None,
        },
        "pascal_yolo": {
            'batch_size': 128,
            'learning_rate': 0.001,
            'training_epochs': 20,
            'global_clipnorm': 10,
            'n_classes': None,
            'optimizer': "adam",
            'val_size': None
        }
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
        model_config['global_clipnorm'] = global_clipnorm if global_clipnorm is not None else model_config['global_clipnorm']

        return model_config
    else:
        raise ValueError(f"Configuration not found for model: {model_name}")
    

# Plot training loss function
def plot_loss(history, model_type: str):
    """
    Plot the training and validation loss over epochs.

    Parameters:
        history (tensorflow.python.keras.callbacks.History): The training history obtained from model.fit.
        model_type: Type of the model(object_detection,classification,segmentation etc.)
    Returns:
        None
    """
    # Get the training and validation loss from the history
    if model_type in ['classification', 'segmentation']:
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

    elif model_type == 'object_detection':
        regression_loss = history.history['box_loss']
        regression_val_loss = history.history['val_box_loss']
        classification_loss = history.history['class_loss']
        classification_val_loss = history.history['val_class_loss']

        epochs = range(1,len(regression_loss) + 1)
        # Create a 1x2 subplot grid
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Bounding Boxes loss
        axs[0].plot(epochs, regression_loss, label='Training Regression Loss')
        axs[0].plot(epochs, regression_val_loss, label='Validation Regression Loss')
        axs[0].set_title('Regression Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Class loss
        axs[1].plot(epochs, classification_loss, label='Training Classification Loss')
        axs[1].plot(epochs, classification_val_loss, label='Validation Classification Loss')
        axs[1].set_title('Classification Loss')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        # Adjust x-axis ticks
        max_ticks = 5
        tick_positions = list(range(1, len(regression_loss) + 1, max(len(regression_loss) // max_ticks, 1)))
        plt.xticks(tick_positions)
    else:
        print('''Please select a valid type of model to 
              plot loss.''')

# Classification confusion matrix plot
def plot_confusion_matrix(model: tf.keras.Model,
                          test_dataset, 
                          class_names: list, 
                          cmap=plt.cm.Blues):
    """
    Plots a confusion matrix for a Keras model on a batched test dataset.

    Parameters:
    - model: Keras model
    - test_dataset: Batched test dataset (tf.data.Dataset or similar)
    - class_names: List of class names
    - normalize: If True, normalize the confusion matrix
    - cmap: Color map for the plot

    Returns:
    - None
    """
    y_true = []
    y_pred = []
    batches_count = len(test_dataset)
    iterator = iter(test_dataset)

    # Iterate over batches to get predictions and true labels
    for i in range(0, batches_count):
        batch_x, batch_y = next(iterator)
        y_pred_batch = np.argmax(model.predict(batch_x, verbose=0), axis=1)
        y_true.extend(batch_y['classes'].numpy())
        y_pred.extend(y_pred_batch)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Get unique class labels
    classes = unique_labels(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(len(class_names), len(class_names)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Visualize object detection predictions from a Keras CV model
def visualize_object_predictions(model: tf.keras.Model, 
                                 dataset, 
                                 bounding_box_format: str, 
                                 class_mapping: dict):
    ''' 
    Visualize object detection predictions from a Keras CV model

    Parameters:
        - model: Keras model
        - dataset: Batched test dataset (tf.data.Dataset or similar)
        - bounding_box_format: Format of bounding boxes
        - class_mapping: Index/Class mapping dictionary
    '''
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images, verbose=0)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
        legend = True
    )


def visualize_segmentation_predictions(dataset: tf.data.Dataset, 
                                       model: tf.keras.Model, 
                                       num_samples:int = 3, 
                                       threshold: float = 0.5):
    """ 
    Visualize original image, mask and predicted segmentation masks from a batched dataset

    Parameters:
       - model: Keras model
       - dataset: Batched test dataset (tf.data.Dataset or similar)
       - num_samples: Number of samples to visualize (range: (1,batch_size))
       - threshold: Value to threshold the predicted masks (range: (0,1))
    """
    samples = dataset.take(1)

    for images, masks in samples:
        for x in range(0, num_samples):
            image_np = images[x].numpy()
            original_mask = masks[x].numpy()

            # Add an extra dimension to match the input shape expected by the model
            image_np = np.expand_dims(image_np, axis=0)

            # Predict the mask using the trained model
            predicted_mask = model.predict(image_np, verbose = 0)

            # Threshold the predicted mask (assuming binary segmentation)
            predicted_mask[predicted_mask >= threshold] = 1
            predicted_mask[predicted_mask < threshold] = 0

            # Plot the original image, original mask, and predicted mask
            plt.figure(figsize=(12, 5))

            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(np.squeeze(image_np[0]))
            plt.title('Original Image')
            plt.axis('off')

            # Original Mask
            plt.subplot(1, 3, 2)
            plt.imshow(original_mask[:, :, 0], cmap='gray')
            plt.title('Original Mask')
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

# Calculate class weights across dataset function
def calculate_average_class_weights(dataset):
    # Function to calculate class weights per batch
    def calculate_class_weights(labels):
        # Flatten the labels and convert them to a NumPy array
        flat_labels = np.concatenate([label.numpy().flatten() for label in labels])

        # Calculate class weights using compute_class_weight
        classes = np.unique(flat_labels)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_labels)

        # Create a dictionary mapping class indices to weights
        class_weight_dict = dict(zip(classes, weights))

        return class_weight_dict

    # Take the training dataset
    iterator = iter(dataset)

    batch_weights = {0: [], 1: []}

    for i in range(len(dataset)):
        _, labels_batch = next(iterator)

        class_weights = calculate_class_weights(labels_batch)

        for key in batch_weights.keys():
            batch_weights[key].append(class_weights.get(key, 0))

    # Calculate average weights across the dataset
    for key in batch_weights.keys():
        batch_weights[key] = sum(batch_weights[key]) / len(batch_weights[key])

    return batch_weights