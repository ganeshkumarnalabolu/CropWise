
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

# 1. Load the dataset
data = pd.read_csv(DATASET_PATH)

# 2. Validate the required columns
features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
target = "label"

required_columns = features + [target]
missing_columns = set(required_columns) - set(data.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

if data[required_columns].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

# 3. Separate inputs and target
X = data[features]
y = data[target]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 5. Train the Random Forest classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# 6. Evaluate the model on unseen test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===== CropWise Model Evaluation =====")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Number of crop classes: {y.nunique()}")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Save the trained model and feature order
joblib.dump(
    {
        "model": model,
        "features": features,
        "classes": model.classes_.tolist(),
    },
    MODEL_PATH,
)

print(f"\nModel saved successfully: {MODEL_PATH}")
