# Brain Tumor Detection System

## Overview
This project aims to develop a computer vision model for the detection of brain tumors using MRI scans. The model utilizes a Convolutional Neural Network (CNN) architecture trained on a dataset of MRI images labeled as containing tumors ("YES") or not containing tumors ("NO").

## Purpose
The detection of brain tumors from MRI scans plays a crucial role in early diagnosis and treatment planning. This project addresses the need for accurate and efficient methods of tumor detection, which can assist healthcare professionals in providing timely medical intervention to patients.

## Model Architecture
The model architecture consists of a CNN with several convolutional and pooling layers followed by fully connected layers. The final layer uses a sigmoid activation function to output a binary classification result.


## Repository Structure
- `README.md`: Provides an overview of the project, setup instructions, and usage guidelines.
- `BrainTumorDetection_CNN.h5`: Trained model file.
- `data_preprocessing.py`: Script for preprocessing MRI images.
- `evaluation.py`: Script for evaluating the model's performance.
- `train.py`: Script for training the model.
- `visualization.py`: Script for visualizing model predictions and performance.
- `documentation`: Additional documentation files (if any).

## Resources
- Dataset: [Link to dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data)

