## Deep Learning Explained: Computer Vision 
This repository contains small tutorials and reference projects around the basic concepts of modern Computer Vision using Deep Learning. A high-level overview of the topics covered and the respective ideal learning outcomes in this course is provided below. 

## Course Content

### Introduction to Computer Vision and Deep Learning Basics
- Gain a fundamental understanding of computer vision and its intersection with deep learning.
- Learn how to create, annotate, and manipulate image datasets.
- Explore the basics of image classification using a simple linear layer.
- Observe how Convolutional Neural Networks (CNNs) interpret image data.
- Get introduced to the concept of data augmentation for enhancing performance of computer vision models.

**Jupyter Notebook**: `basics/introduction.ipynb`

### Getting Started with Image Classification
1. **Image Preprocessing:** Explore techniques to prepare and preprocess image data for model training.
2. **Building CNN Models:** Learn how to build your first Deep Learning model for image classification with Tensorflow and Keras.
3. **Train your CNN model:** Train your first CNN model, explore different training callbacks, and understand the role of the loss function.
4. **Evaluating Model Performance:** Learn how to evaluate the performance of image classification models using metrics like accuracy, precision, and recall and visualize the confusion matrix.
5. **Test performance on unseen data:** Test the performance of your model on data unkown to the model.

**Jupyter Notebook**: `image_classifier/classification.ipynb`

### Object Detection with YoloV8
1. **Introduction to Object Detection:** Dive into the advanced Computer Vision task of object detection by training your first state-of-the-art model.
2. **Introduction to YoloV8:** Familiarize yourself with the YoloV8 architecture and it's implementation in Keras using the Keras-CV API.
3. **Object Detection data pre-processing:** Get hands-on experience on how to prepare a dataset to train an object detection Deep Learning model.
4. **Training YoloV8:** Experiment with different training configurations and strategies to optimize the YoloV8 model's performance.
5. **Evaluating Object Detection Performance:** Understand evaluation metrics for object detection, such as mean Average Precision (mAP).
6. **Test performance on unseen data:** Test the performance of your model on data unkown to the model.

**Jupyter Notebook**: `object_detector/object_detection.ipynb`

### Semantic Segmentation with UNet
1. **Introduction to Semantic Segmentation:** Explore the advanced Computer Vision task of semantic segmentation.
2. **UNet Architecture:** Understand the U-Net architecture and it's implementation with the Keras Functional API.
3. **Data Preparation:** Learn how to prepare a dataset of images and segmentation mask annotations for a semantic segmentation model.
4. **Transfer Learning:** Understand the concept of Transfer Learning by leveraging a pre-trained feature extraction model as basis to train your model.
5. **Training and Evaluation:** Train the UNet model, explore more training callback functions and evaluate its performance using metrics like Intersection over Union (IoU).

**Jupyter Notebook**: `semantic_segmentation/semantic_segmentation.ipynb`

### Use AWS Sagemaker to train your model in the cloud
**Familiarize yourself with the AWS environment**: Familiarize yourself with tasks like data storing on S3 buckets, the concepts of users,roles and policies in AWS and learn how to set up and run a model training job using AWS Sagemaker.

**NOTE**: Before running any of the following scripts make sure to add your authentication credentials and following the instructions provided in each script.

**NOTE**: Remember to delete resources and keep track of how much you are charged for anything related to AWS Services, as extensively using cloud resources can be quite costly!

**Upload dataset to an S3 Bucket script**: `sagemaker/upload_dataset.py`

**Training job entrypoint script**: `sagemaker/train.py` 

**Train a deep learning model with Sagemaker script**: `sagemaker/run_sagemaker_job.py`


### Extras

**Utility functions for all modules**: `helper.py` \
**Google Collaboratory supported notebooks**: `google_collab/`


