from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load datasets
pesticides_df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/Crop Yield Prediction/datasets/pesticides.csv')
rainfall_df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/Crop Yield Prediction/datasets/rainfall.csv')
temp_df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/Crop Yield Prediction/datasets/temp.csv')
yield_full_df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/Crop Yield Prediction/datasets/yield_df.csv')

# Standardize column names
rainfall_df.rename(columns=lambda x: x.strip(), inplace=True)
yield_full_df.rename(columns=lambda x: x.strip(), inplace=True)
pesticides_df.rename(columns=lambda x: x.strip(), inplace=True)
temp_df.rename(columns=lambda x: x.strip(), inplace=True)

# Rename columns for consistency
rainfall_df.rename(columns={"Area": "country", "Year": "year", "average_rain_fall_mm_per_year": "rainfall"}, inplace=True)
yield_full_df.rename(columns={"Area": "country", "Year": "year", "hg/ha_yield": "yield"}, inplace=True)
pesticides_df.rename(columns={"Area": "country", "Year": "year", "Value": "pesticides_tonnes"}, inplace=True)
temp_df.rename(columns={"year": "year", "country": "country", "avg_temp": "temperature"}, inplace=True)

# Print column names for debugging
print("Initial Columns in Pesticides Dataset:", pesticides_df.columns)
print("Processed Columns in Pesticides Dataset:", pesticides_df.columns)

# Merge datasets
merged_df = yield_full_df.merge(pesticides_df[['country', 'year', 'pesticides_tonnes']], on=['country', 'year'], how='left')
merged_df = merged_df.merge(rainfall_df[['country', 'year', 'rainfall']], on=['country', 'year'], how='left')
merged_df = merged_df.merge(temp_df[['country', 'year', 'temperature']], on=['country', 'year'], how='left')

# Ensure correct column selection
if 'pesticides_tonnes_x' in merged_df.columns:
    merged_df.rename(columns={'pesticides_tonnes_x': 'pesticides_tonnes'}, inplace=True)
elif 'pesticides_tonnes_y' in merged_df.columns:
    merged_df.rename(columns={'pesticides_tonnes_y': 'pesticides_tonnes'}, inplace=True)

# Print final merged DataFrame columns
print("Final Merged DataFrame Columns:", merged_df.columns)

# Drop unnecessary columns and clean data
merged_df.dropna(inplace=True)

# Select features and target variable
X = merged_df[['rainfall', 'pesticides_tonnes', 'temperature']]
y = merged_df['yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        rainfall = float(data.get('average_rainfall_mm', 0))
        pesticides = float(data.get('pesticides_tonnes', 0))
        temperature = float(data.get('avg_temp', 0))

        input_features = np.array([[rainfall, pesticides, temperature]])
        predicted_yield_hg_ha = model.predict(input_features)[0]  # hg/ha

        # Convert to kg/ha and tons/ha
        predicted_yield_kg_ha = predicted_yield_hg_ha * 10
        predicted_yield_tons_ha = predicted_yield_hg_ha / 100

        return jsonify({
            'predicted_yield_hg_per_ha': predicted_yield_hg_ha,
            'predicted_yield_kg_per_ha': predicted_yield_kg_ha,
            'predicted_yield_tons_per_ha': predicted_yield_tons_ha,
            'unit': 'per hectare'
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
