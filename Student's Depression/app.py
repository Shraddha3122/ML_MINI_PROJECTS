import pandas as pd
import joblib
import numpy as np
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model, scaler, and encoders
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load correct feature names used during training
with open("feature_names.json", "r") as f:
    correct_features = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON input
        df = pd.DataFrame([data])  # Convert input to DataFrame

        # Encode categorical features safely (handle unseen labels)
        for col in label_encoders:
            if col in df.columns:
                le = label_encoders[col]
                known_classes = set(le.classes_)  # Get known values from training
                df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])  # Replace unknown values
                df[col] = le.transform(df[col])  # Encode categorical column

        # Ensure all required features exist in the input
        for col in correct_features:
            if col not in df.columns:
                df[col] = 0  # Assign default value if missing

        df = df[correct_features]  # Reorder columns correctly
        df_scaled = scaler.transform(df)  # Scale the input
        prediction = model.predict(df_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "message": "1 indicates depression risk, 0 indicates no risk."
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
