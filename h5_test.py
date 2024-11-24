import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

nn_model = load_model("model/neural.h5")
meta_model = joblib.load("model/optim.pkl")
scaler = joblib.load("model/scaler.pkl")
rf_model = joblib.load("model/rf_model.pkl")
sleep_disorder_encoder = joblib.load("model/sleep_disorder_encoder.pkl")
top_feature_names = joblib.load("model/top_feature_names.pkl")

# Test data 94 apnea, 98 nan, 6 Insomia
test_data = {
    "BMI Category": 0,
    "Age": 36,
    "Sleep Duration": 7.1,
    "Physical Activity Level": 60,
    "Daily Steps": 7000,
}

# test_df = pd.DataFrame([test_data])

# test_scaled = scaler.transform(test_df)

# test_top_features = test_df[top_feature_names].values

# nn_predictions = nn_model.predict(test_scaled)
# nn_predicted_class = np.argmax(nn_predictions, axis=1)[0]
# nn_predicted_label = sleep_disorder_encoder.inverse_transform([nn_predicted_class])[0]

# meta_features = np.hstack([nn_predictions, test_top_features])

# meta_predicted_class = meta_model.predict(meta_features)[0]
# meta_predicted_label = sleep_disorder_encoder.inverse_transform([meta_predicted_class])[0]

# nn_confidence = nn_predictions[0, nn_predicted_class]
# if nn_confidence > 0.7:
#     final_prediction = nn_predicted_label  # Favor NN when confidence is high
# else:
#     final_prediction = meta_predicted_label  # Default to Meta-Model otherwise


# print(f"Raw NN Predictions: {nn_predictions}")
# print(f"NN Confidence for {nn_predicted_label}: {nn_confidence:.2f}")
# print(f"Meta-Model Features: {meta_features}")
# print(f"NN Predicted: {nn_predicted_label}")
# print(f"Meta Predicted: {meta_predicted_label}")
# print(f"Final Prediction: {final_prediction}")



# top_features = joblib.load("/content/top_feature_names.pkl")

# test_df = pd.DataFrame([test_data])[top_features]

# print("Features in test_df:", test_df.columns.tolist())
# print("Features scaler was fit on:", scaler.feature_names_in_.tolist())



test_df = pd.DataFrame([test_data])
test_df = test_df[top_feature_names]

test_scaled = scaler.transform(test_df)

nn_predictions = nn_model.predict(test_scaled)
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
if pd.isna(final_prediction):
    final_prediction = "Healthy"

print(f"NN Predicted: {nn_predicted_label}")
print(f"RF Predicted: {rf_predicted_label}")
print(f"Meta Predicted: {meta_predicted_label}")
print(f"Final Prediction: {final_prediction}")
