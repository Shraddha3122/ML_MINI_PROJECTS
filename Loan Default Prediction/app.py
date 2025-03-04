import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load dataset
df = pd.read_csv("D:/WebiSoftTech/ML_MINI_PROJECTS/Loan Default Prediction/Loan_default.csv")

# Drop unique identifier columns
if "LoanID" in df.columns:
    df.drop(columns=["LoanID"], inplace=True)

# Handle missing values
df.dropna(subset=["Default"], inplace=True)  # Drop rows where target is missing
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numerical NaN values

# Define features and target
X = df.drop(columns=["Default"])
y = df["Default"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Handle categorical encoding
high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 50]
low_cardinality_cols = list(set(categorical_cols) - set(high_cardinality_cols))

# One-hot encode low-cardinality categorical columns
X = pd.get_dummies(X, columns=low_cardinality_cols, drop_first=True)

# Convert high-cardinality categorical columns to category codes
for col in high_cardinality_cols:
    X[col] = X[col].astype("category").cat.codes

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear')
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores[name] = roc_auc_score(y_test, y_pred)

# Select best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

# Save the best model
joblib.dump(best_model, "best_model.pkl")

# Load the best model
model = joblib.load("best_model.pkl")

# Flask app
app = Flask(__name__)

@app.route('/all_scores', methods=['GET'])
def all_scores():
    return jsonify(scores)

@app.route('/best_model', methods=['GET'])
def best_model_info():
    return jsonify({"best_model": best_model_name, "score": scores[best_model_name]})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        feature_names = joblib.load("feature_names.pkl")
        input_df = pd.DataFrame([data])

        # Ensure missing columns from one-hot encoding are set to 0
        missing_cols = set(feature_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # Assign missing features a default value

        # Reorder columns to match training set
        input_df = input_df[feature_names]
        
        # Convert to NumPy array
        features = input_df.to_numpy()

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        return jsonify({
            "prediction": "Default" if prediction[0] == 1 else "No Default",
            "probability": float(probability[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
