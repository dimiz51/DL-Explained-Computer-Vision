# Imports
import sagemaker
from sagemaker.tensorflow import TensorFlow
import argparse

if __name__ == "__main__":
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args, _ = parser.parse_known_args()


    # Define the S3 bucket and prefix where your data is stored
    bucket_name = 'cifar10-dlexplained'
    s3_prefix = 'pre-processed'

    # Define your SageMaker role
    role = 'arn:aws:iam::874208852027:role/SagemakerFullAccess'

    # Set up paths for train and validation data
    s3_train_path = f's3://{bucket_name}/{s3_prefix}/train'
    s3_val_path = f's3://{bucket_name}/{s3_prefix}/validation'

    # Create a TensorFlow Estimator for SageMaker
    estimator = TensorFlow(
        entry_point='train.py',  # Specify the script that contains your training code
        role=role,
        instance_count=1,
        instance_type='ml.m5.4xlarge',  # Choose an appropriate instance type
        framework_version='2.13.0',
        py_version='py310',
        script_mode=True,
        hyperparameters={
        'epochs': args.epochs
        }
    )

    # Set up the input data channels
    train_data = sagemaker.inputs.TrainingInput(
        s3_train_path,
        distribution='ShardedByS3Key',
        content_type='application/tfrecord',
        s3_data_type='S3Prefix',
    )

    val_data = sagemaker.inputs.TrainingInput(
        s3_val_path,
        distribution='ShardedByS3Key',
        content_type='application/tfrecord',
        s3_data_type='S3Prefix',
    )

    # Train the model on SageMaker
    estimator.fit({'train': train_data, 'validation': val_data})



# Steps
# Create a role named SagemakerFullAccess with read/write access to the previously created bucket to save tfrecords
# Set training script to load and parse the tf records and create/train model
# Run