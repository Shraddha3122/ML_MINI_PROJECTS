import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("D:/WebiSoftTech/ML_MINI_PROJECTS/Student's Depression/Student Depression Dataset.csv")

# Drop unnecessary columns
df.drop(columns=["id", "City", "Profession"], errors='ignore', inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
df[df.columns] = imputer.fit_transform(df)

# Encode categorical variables
label_encoders = {}
categorical_cols = ["Gender", "Sleep Duration", "Dietary Habits", "Degree", 
                    "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoders for inference

# Define features and target
X = df.drop(columns=["Depression"], errors='ignore')
y = df["Depression"].astype(int)

# ** Save feature order for inference **
feature_names = list(X.columns)
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Optimized Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=7)
gb_model.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

# Train XGBoost Model
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# Fix XGBoost for Stacking (Wrap in Pipeline)
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Ensures consistent scaling
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, use_label_encoder=False, eval_metric='logloss'))
])

# Train Stacking Model
stacked_model = StackingClassifier(
    estimators=[('xgb', xgb_pipeline), ('rf', RandomForestClassifier(n_estimators=300))],
    final_estimator=LogisticRegression()
)
stacked_model.fit(X_train, y_train)
stacked_acc = accuracy_score(y_test, stacked_model.predict(X_test))
print(f"Stacked Model Accuracy: {stacked_acc:.4f}")

# Train ANN Model
ann_model = Sequential([
    Input(shape=(X_train.shape[1],)),  # FIXED input layer
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

ann_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), 
              callbacks=[early_stopping], verbose=1)

_, ann_acc = ann_model.evaluate(X_test, y_test)
print(f"ANN Accuracy: {ann_acc:.4f}")

# Save the best performing model
best_model = max([(gb_model, gb_acc), (xgb_model, xgb_acc), (stacked_model, stacked_acc)], key=lambda x: x[1])[0]
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")  # Save encoders
ann_model.save("ann_model.h5")

print("Best model and ANN saved successfully!")
