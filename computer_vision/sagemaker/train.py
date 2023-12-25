# Imports
import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os

# Import SageMaker Training Toolkit
from sagemaker_training import environment

# Default model config
config = {"batch_size": 128,
         "training_epochs": 30,
         "n_classes": 10,
         "learning_rate": 0.001,
         "optimizer": "adam",
         "val_size": 0.2,
         "global_clipnorm": None}

'''Create a random seed generator for randomized TF ops'''
rng = tf.random.Generator.from_seed(123, alg='philox')

# Create a model using the Keras Sequential API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_cifar10_model(model_config: dict):
    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', data_format='channels_last', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(model_config['n_classes'], activation = None)
    ], name='cifar10_model')

    # Compile the model
    optimizer = Adam(learning_rate=model_config['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='Accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(name='TopKAccuracy')
        ]
    )

    return model

# Function to train the model
def train_model(model, ds_train, ds_val, epochs):
    # Set Early Stopping strategy after 5 epochs of no improvement for validation set
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(ds_train, 
                        epochs= int(epochs), 
                        validation_data= ds_val, 
                        callbacks = [callback],
                        verbose = 1)
    return history

# Pre-processing
def normalize_image(image: tf.Tensor,label: tf.Tensor)-> (tf.Tensor, tf.Tensor):
    '''Define a normalization function that rescales 
    the pixel values from [0,255] uint8 to float32 [0,1]
    '''
    return tf.cast(image, tf.float32) / 255., label

# Augmentations
def augment_image(image_label: tuple, seed)-> (tf.Tensor, tf.Tensor):
    '''Apply basic augmentations on the training dataset samples
    in order to induce extra variance to our dataset. This can help
    our model generalize in a better way. Augmentations applied are 
    random horizontal flip, random crop and random rotation by 45 degrees.'''

    image, label = image_label
    new_seed = tf.random.split(seed, num=1)[0, :]
    image = tf.image.stateless_random_flip_left_right(image, new_seed)

    angle = tf.random.uniform(shape=(), minval=-45, maxval=45, dtype=tf.float32)
    image = tf.image.rot90(image, k=tf.cast(angle / 90, dtype=tf.int32))

    image = tf.image.stateless_random_crop(value= image, size= (32,32,3), seed= new_seed)

    return image, label


def random_wrapper(image: tf.Tensor, label: tf.Tensor)-> (tf.Tensor, tf.Tensor):
    '''Wrapper function for our augmentations to generate a new random 
    seed on each call. This way we can indeed have random augmentations 
    for each sample.'''

    seed = rng.make_seeds(2)[0]
    image, label = augment_image((image, label), seed)
    return image,label


def load_data(train_path:str,
              val_path: str):
    """Load the tfrecord objects from the S3 bucket"""
    # Load data from SageMaker environment
    ds_train = tf.data.TFRecordDataset(filenames=[f'{train_path}/{file}' for file in os.listdir(train_path)])
    ds_val = tf.data.TFRecordDataset(filenames=[f'{val_path}/{file}' for file in os.listdir(val_path)])

    return ds_train, ds_val



# Parsing function to parse the TFRecord
def _parse_function(example_proto):
    # Define the feature description for parsing the TFRecord
    feature_description = {'image': tf.io.FixedLenFeature([], tf.string),
                           'label': tf.io.FixedLenFeature([], tf.int64)
                        }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed_example['image'], channels=3)
    label = parsed_example['label']
    return image, label


if __name__ == "__main__":
    # Set up SageMaker environment
    train_env = environment.Environment()

    # Load data from SageMaker environment
    s3_train_path = train_env.channel_input_dirs["train"]
    s3_val_path = train_env.channel_input_dirs["validation"]
    print("S3 Train Path:", s3_train_path)
    print("S3 Validation Path:", s3_val_path)

    ds_train, ds_val = load_data(s3_train_path, s3_val_path)

    # Calculate the number of examples in the training set
    num_train_examples = sum(1 for _ in ds_train)

    # Calculate the number of examples in the validation set
    num_val_examples = sum(1 for _ in ds_val)

    # Print the number of examples in the training and validation sets
    print("Number of images in training set:", num_train_examples)
    print("Number of images in validation set:", num_val_examples)

    # Apply the parsing function to the datasets
    ds_val = ds_val.map(_parse_function)
    ds_train = ds_train.map(_parse_function)


    # Pipelining pre-processing, augmentations, batching and caching of the training dataset.
    ds_train = ds_train.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(num_train_examples, reshuffle_each_iteration=True)
    ds_train = ds_train.map(random_wrapper, 
                        num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)

    # Prepare the validation set
    ds_val = ds_val.cache()
    ds_val = ds_val.shuffle(num_val_examples, reshuffle_each_iteration=True)
    ds_val = ds_val.batch(config['batch_size']).map(normalize_image)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    # How to read an argument from the estimator if it's provided
    epochs = int(train_env.hyperparameters.get('epochs'))

    # Create model
    model = create_cifar10_model(config)

    # Train the model
    history = train_model(model, ds_train, ds_val, epochs)

    # Save the model
    model.save(f'{train_env.model_dir}/cifar10_model')
