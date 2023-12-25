# Imports
import sys
sys.path.append('../')
import tensorflow as tf
import tensorflow_datasets as tfds
from sagemaker.session import Session

import boto3
from botocore.exceptions import NoCredentialsError

# Steps
# Create AWS account
# Create a bucket named cifar10-dlexplained
# Create a user and give access to read/write to that bucket to write tfrecords
# Replace key_id and secret access key in "main" with your own
# Run this


def configure_aws_credentials(aws_access_key_id, aws_secret_access_key, region_name):
    """Configure AWS credentials programmatically."""
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        s3_client = session.client('s3')
        return s3_client
    except NoCredentialsError:
        print("Credentials not available or incorrect.")
        return None


# Default config
model_config = {"batch_size": 128,
                "training_epochs": 30,
                "n_classes": 10,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "val_size": 0.2,
                "global_clipnorm": None}

if __name__ == "__main__":

    # Replace the placeholders with your actual AWS credentials
    aws_access_key_id = "<YOUR_ACCESS_KEY_ID>"
    aws_secret_access_key = "<YOUR_SECRET_ACCESS_KEY>"
    aws_region = "eu-north-1" 

    # Configure AWS credentials
    s3_client = configure_aws_credentials(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )


    # Load the CIFAR-10 dataset from the local cache
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    print(ds_info)

    # Generate the validation set from the training set
    validation_size = int(model_config['val_size'] * ds_info.splits['train'].num_examples) 

    # Create the validation dataset
    ds_val = ds_train.take(validation_size)
    ds_train = ds_train.skip(validation_size)

    from tensorflow.train import BytesList, Int64List
    from tensorflow.train import Feature, Features, Example

    # Define the paths to save the TFRecord files
    train_tfrecord_path = "./dataset_preprocessed/train.tfrecord"
    val_tfrecord_path = "./dataset_preprocessed/validation.tfrecord"

    # Create TFRecordWriter for training set
    with tf.io.TFRecordWriter(train_tfrecord_path) as writer:
        for image, label in ds_train:
            # Convert image to bytes
            image_bytes = tf.io.encode_jpeg(image)
        
            # Create an example explaining the features to write
            example = Example(
                features=Features(
                    feature={
                        'image': Feature(bytes_list=BytesList(value=[image_bytes.numpy()])),
                        'label': Feature(int64_list=Int64List(value=[label.numpy()]))
                    }
                )
            )
        
            # Serialize sample and write to TFRecord file
            writer.write(example.SerializeToString())

    # Create TFRecordWriter for validation set
    with tf.io.TFRecordWriter(val_tfrecord_path) as writer:
        for image, label in ds_val:
            # Convert image to bytes
            image_bytes = tf.io.encode_jpeg(image)
        
            # Create an example explaining the features to write
            example = Example(
                features=Features(
                    feature={
                        'image': Feature(bytes_list=BytesList(value=[image_bytes.numpy()])),
                        'label': Feature(int64_list=Int64List(value=[label.numpy()]))
                    }
                )
            )
        
            # Serialize sample and write to TFRecord file
            writer.write(example.SerializeToString())

    # Set up AWS and upload pre-processed datasets
    bucket_name = 'cifar10-dlexplained'
    s3_prefix = 'pre-processed'

    # Check AWS access and upload datasets
    if s3_client:
        # Upload pre-processed datasets to S3
        s3_train_path = Session().upload_data(path=train_tfrecord_path, bucket=bucket_name, key_prefix=s3_prefix+'/train')
        s3_val_path = Session().upload_data(path=val_tfrecord_path, bucket=bucket_name, key_prefix=s3_prefix+'/validation')

        print(f"Training set uploaded to: {s3_train_path}")
        print(f"Validation set uploaded to: {s3_val_path}")



