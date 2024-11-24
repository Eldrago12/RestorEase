# Predicting Sleep Disorders with Neural Networks and Meta-Model Integration

This project is an advanced machine learning pipeline to predict sleep disorders like Insomnia, Sleep Apnea, or Healthy status using a combination of Neural Networks, Random Forest, and a LightGBM meta-model. The pipeline has been optimized for accuracy, achieving an outstanding 90.91% accuracy, surpassing most research papers using traditional models like Random Forest and XGBoost.

## Table of Contents

Introduction
Features
Dataset Preprocessing
Model Building
Training Results
Testing Pipeline
Deployment
Conclusion
Usage

## Introduction

Sleep disorders affect millions of individuals worldwide. Early and accurate detection can improve quality of life. This project builds a predictive model using a hybrid approach that outperforms traditional machine learning techniques.

## Features
- Dataset preprocessing with SMOTE for class balancing.
- Noise addition via data augmentation.
- Feature selection using Random Forest.
- Neural Network with:

  - **256 neurons per layer**
  - **Dropout regularization**
  - **ReLU activation**

## Meta-model (LightGBM) combining:

  - **Neural Network predictions**
  - **Random Forest probabilities**
  - **Top 5 features**


## Dataset Preprocessing

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


## Model Building

1. **Random Forest for Feature Selection**

- Identified the top 5 features:
  
  - **BMI Category**
  - **Age**
  - **Sleep Duration**
  - **Physical Activity Level**
  - **Daily Steps**


2. **Neural Network**

- Architecture:
  
  - Input layer with 256 neurons.
  - Hidden layers with Dropout and ReLU activation.
  - Output layer with softmax activation.

- Optimization:
  
  - Loss: Sparse Categorical Crossentropy.
  - Optimizer: RMSProp with learning rate 1e-4.
  - Regularization: L2.

- Training:
  
  - Class weights to handle imbalance.
  - Early stopping and learning rate reduction.


3. **LightGBM Meta-Model**

- Combined:

 - **Neural Network predictions**
 - **Random Forest probabilities**
 - **Top 5 features**

- Regularization parameters:
  
  - **reg_alpha=2.0, reg_lambda=2.0**


## Training Results:

- Neural Network Accuracy: 90.91%
- Meta-Model Accuracy: 90.91%

## Testing Pipeline

  ```bash
  import numpy as np
  import pandas as pd
  import joblib
  from tensorflow.keras.models import load_model
  
  nn_model = load_model("neural.h5")
  meta_model = joblib.load("optim.pkl")
  scaler = joblib.load("scaler.pkl")
  rf_model = joblib.load("rf_model.pkl")
  sleep_disorder_encoder = joblib.load("sleep_disorder_encoder.pkl")
  top_feature_names = joblib.load("top_feature_names.pkl")
  
  test_data = {
      "BMI Category": 2,
      "Age": 35,
      "Sleep Duration": 7.4,
      "Physical Activity Level": 60,
      "Daily Steps": 3300,
  }
  
  test_df = pd.DataFrame([test_data])[top_feature_names]
  test_scaled = scaler.transform(test_df)
  
  nn_predictions = nn_model.predict(test_scaled)
  nn_predicted_class = np.argmax(nn_predictions, axis=1)[0]
  nn_predicted_label = sleep_disorder_encoder.inverse_transform([nn_predicted_class])[0]
  
  rf_probs = rf_model.predict_proba(test_scaled)
  
  meta_features = np.hstack([nn_predictions, rf_probs, test_scaled])
  meta_predicted_class = meta_model.predict(meta_features)[0]
  meta_predicted_label = sleep_disorder_encoder.inverse_transform([meta_predicted_class])[0]
  
  nn_confidence = nn_predictions[0, nn_predicted_class]
  final_prediction = nn_predicted_label if nn_confidence > 0.7 else meta_predicted_label
  
  print(f"NN Predicted: {nn_predicted_label}")
  print(f"Meta Predicted: {meta_predicted_label}")
  print(f"Final Prediction: {final_prediction}")
  ```

## Deployment

 - **Option 1: Flask App**

   Install Dependencies:

   ```bash
   pip install flask tensorflow joblib pandas numpy lightgbm
   ```

   Create app.py:

   ```bash
   from flask import Flask, request, jsonify
   import numpy as np
   import pandas as pd
   import joblib
   from tensorflow.keras.models import load_model
    
   app = Flask(__name__)
    
    # Load Models and Artifacts
   nn_model = load_model("new_neural.h5")
   meta_model = joblib.load("optim.pkl")
   scaler = joblib.load("scaler.pkl")
   rf_model = joblib.load("rf_model.pkl")
   sleep_disorder_encoder = joblib.load("sleep_disorder_encoder.pkl")
   top_feature_names = joblib.load("top_feature_names.pkl")
    
   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json
       test_df = pd.DataFrame([data])[top_feature_names]
       test_scaled = scaler.transform(test_df)
    
       nn_predictions = nn_model.predict(test_scaled)
       nn_predicted_class = np.argmax(nn_predictions, axis=1)[0]
       nn_predicted_label = sleep_disorder_encoder.inverse_transform([nn_predicted_class])[0]
  
       rf_probs = rf_model.predict_proba(test_scaled)
       meta_features = np.hstack([nn_predictions, rf_probs, test_scaled])
       meta_predicted_class = meta_model.predict(meta_features)[0]
       meta_predicted_label = sleep_disorder_encoder.inverse_transform([meta_predicted_class])[0]
    
       nn_confidence = nn_predictions[0, nn_predicted_class]
       final_prediction = nn_predicted_label if nn_confidence > 0.7 else meta_predicted_label
    
       return jsonify({
           "NN Predicted": nn_predicted_label,
           "Meta Predicted": meta_predicted_label,
           "Final Prediction": final_prediction
        })
    
    if __name__ == '__main__':
        app.run(debug=True)
    ```

    Run the App:

    ```bash
    python app.py
    ```

  - **Option 2: AWS Lambda**

    Convert the TensorFlow model to TFlite:

    ```bash
    pip install tflite-runtime
    
    python -m tflite.convert --saved-model neural.h5 --output neural.tflite
    ```

    Package Artifacts:

    ```bash
    zip -r deployment.zip neural.tflite optim.pkl scaler.pkl rf_model.pkl sleep_disorder_encoder.pkl top_feature_names.pkl lambda_function.py
    ```

    Deploy to AWS Lambda:

    - Use Lambda Layers for dependencies
    - Set up an API Gateway for inference


## Conclusion

This project demonstrates the power of combining Neural Networks, Random Forest, and a LightGBM meta-model to achieve state-of-the-art results in sleep disorder prediction. Its flexibility for deployment ensures usability across platforms.
