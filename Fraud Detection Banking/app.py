import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
from flask import Flask, request, jsonify

# Load dataset
df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/Fraud Detection Banking/creditcard.csv')

# Feature selection
X = df.drop(columns=['Class'])
y = df['Class']

# Normalize all features
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scale entire dataset

# Handle imbalance
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['features']).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Scale entire input
    prediction = model.predict(input_data)
    return jsonify({'fraudulent': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
