from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": ["http://localhost:5500", "http://127.0.0.1:5500", "null"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

try:
    logger.info("Loading models...")
    h5_model = load_model("model/neural.h5")
    meta_model = joblib.load("model/optim.pkl")
    scaler = joblib.load("model/scaler.pkl")
    rf_model = joblib.load("model/rf_model.pkl")
    sleep_disorder_encoder = joblib.load("model/sleep_disorder_encoder.pkl")
    top_feature_names = joblib.load("model/top_feature_names.pkl")
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    logger.info(f"Received {request.method} request")
    logger.info(f"Request headers: {request.headers}")

    if request.method == "OPTIONS":
        logger.info("Handling OPTIONS request")
        response = jsonify(success=True)
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    try:
        logger.info("Parsing request data")
        test_data = request.get_json()
        logger.info(f"Received data: {test_data}")

        logger.info("Converting to DataFrame")
        test_df = pd.DataFrame([test_data])
        logger.info(f"Expected features: {top_feature_names}")
        logger.info(f"Received features: {test_df.columns.tolist()}")

        missing_features = set(top_feature_names) - set(test_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        test_df = test_df[top_feature_names]
        logger.info("DataFrame created successfully")

        logger.info("Scaling data")
        test_scaled = scaler.transform(test_df)

        logger.info("Making NN predictions")
        nn_predictions = h5_model.predict(test_scaled)
        nn_predicted_class = np.argmax(nn_predictions, axis=1)[0]
        nn_predicted_label = sleep_disorder_encoder.inverse_transform([nn_predicted_class])[0]
        nn_confidence = nn_predictions[0, nn_predicted_class]

        logger.info("Making RF predictions")
        rf_probs = rf_model.predict_proba(test_scaled)
        rf_predicted_class = np.argmax(rf_probs, axis=1)[0]
        rf_predicted_label = sleep_disorder_encoder.inverse_transform([rf_predicted_class])[0]

        logger.info("Making Meta-Model predictions")
        meta_features = np.hstack([nn_predictions, rf_probs, test_scaled])
        meta_predicted_class = meta_model.predict(meta_features)[0]
        meta_predicted_label = sleep_disorder_encoder.inverse_transform([meta_predicted_class])[0]

        if nn_confidence > 0.7:
            final_prediction = nn_predicted_label
        else:
            final_prediction = meta_predicted_label

        if pd.isna(final_prediction):
            logger.warning("Final prediction is NaN. Setting to 'Healthy'")
            final_prediction = "Healthy"

        response = {
            "Final Prediction": final_prediction
        }

        logger.info(f"Sending response: {response}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "message": "An error occurred during prediction."
        }), 500

if __name__ == "__main__":
    logger.info(f"Available features: {top_feature_names}")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
