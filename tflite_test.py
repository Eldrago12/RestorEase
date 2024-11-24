import numpy as np
import pandas as pd
import joblib
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="model/neural.tflite")
meta_model = joblib.load("model/optim.pkl")
scaler = joblib.load("model/scaler.pkl")
rf_model = joblib.load("model/rf_model.pkl")
sleep_disorder_encoder = joblib.load("model/sleep_disorder_encoder.pkl")
top_feature_names = joblib.load("model/top_feature_names.pkl")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test data 94 apnea, 98 nan, 6 Insomia
test_data = {
    "BMI Category": 2,
    "Age": 28,
    "Sleep Duration": 5.9,
    "Physical Activity Level": 30,
    "Daily Steps": 3000,
}

test_df = pd.DataFrame([test_data])
test_df = test_df[top_feature_names]

test_scaled = scaler.transform(test_df)

input_data = test_scaled.astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

nn_predictions = interpreter.get_tensor(output_details[0]['index'])

nn_predicted_class = np.argmax(nn_predictions, axis=1)[0]
nn_predicted_label = sleep_disorder_encoder.inverse_transform([nn_predicted_class])[0]
nn_confidence = nn_predictions[0, nn_predicted_class]

rf_probs = rf_model.predict_proba(test_scaled)
rf_predicted_class = np.argmax(rf_probs, axis=1)[0]
rf_predicted_label = sleep_disorder_encoder.inverse_transform([rf_predicted_class])[0]

meta_features = np.hstack([nn_predictions, rf_probs, test_scaled])

meta_predicted_class = meta_model.predict(meta_features)[0]
meta_predicted_label = sleep_disorder_encoder.inverse_transform([meta_predicted_class])[0]

if nn_confidence > 0.7:
    final_prediction = nn_predicted_label
else:
    final_prediction = meta_predicted_label

# Output
print(f"NN Predicted: {nn_predicted_label}")
print(f"RF Predicted: {rf_predicted_label}")
print(f"Meta Predicted: {meta_predicted_label}")
print(f"Final Prediction: {final_prediction}")
