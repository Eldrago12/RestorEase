#Predicting Sleep Disorders with Neural Networks and Meta-Model Integration

This project is an advanced machine learning pipeline to predict sleep disorders like Insomnia, Sleep Apnea, or Healthy status using a combination of Neural Networks, Random Forest, and a LightGBM meta-model. The pipeline has been optimized for accuracy, achieving an outstanding 90.91% accuracy, surpassing most research papers using traditional models like Random Forest and XGBoost.

##Table of Contents

Introduction
Features
Dataset Preprocessing
Model Building
Training Results
Testing Pipeline
Deployment
Conclusion
Usage

##Introduction

Sleep disorders affect millions of individuals worldwide. Early and accurate detection can improve quality of life. This project builds a predictive model using a hybrid approach that outperforms traditional machine learning techniques.

##Features
- Dataset preprocessing with SMOTE for class balancing.
- Noise addition via data augmentation.
- Feature selection using Random Forest.
- Neural Network with:

  -**256 neurons per layer**
  -**Dropout regularization**
  -**ReLU activation**

##Meta-model (LightGBM) combining:

  -**Neural Network predictions**
  -**Random Forest probabilities**
  -**Top 5 features**


##Dataset Preprocessing

    1. Dataset Columns:

    - BMI Category, Age, Sleep Duration, Physical Activity Level, Daily Steps, Heart Rate, Stress Level, Quality of Sleep, and Gender.

    2. Steps:

    - Handled missing values using column means
    
    - Encoded categorical features:
    
      - BMI Category: Normal (0), Overweight (1), Obese (2).
      - Gender: Encoded with LabelEncoder.

    - Target variable (Sleep Disorder) mapping:
    
      - Healthy (0), Insomnia (1), Sleep Apnea (2).


    3. Feature Standardization:
    
      - Applied StandardScaler to normalize numerical features.

    4. Class Balancing with SMOTE:
    
      - Ensured equal representation for all classes.

    5. Data Augmentation:
    
      - Added Gaussian noise for training robustness.


Model Building
1. Random Forest for Feature Selection
Identified the top 5 features:
BMI Category
Age
Sleep Duration
Physical Activity Level
Daily Steps


2. Neural Network
Architecture:
Input layer with 256 neurons.
Hidden layers with Dropout and ReLU activation.
Output layer with softmax activation.

Optimization:
Loss: Sparse Categorical Crossentropy.
Optimizer: RMSProp with learning rate 1e-4.
Regularization: l2.

Training:
Class weights to handle imbalance.
Early stopping and learning rate reduction.


3. LightGBM Meta-Model
Combined:
Neural Network predictions.
Random Forest probabilities.
Top 5 features.

Regularization parameters:
reg_alpha=2.0, reg_lambda=2.0.


Training Results
Neural Network Accuracy: 90.91%.
Meta-Model Accuracy: 90.91%.
