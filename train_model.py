import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load dataset
data = pd.read_csv(BASE_DIR / "mobile_price.csv")

# Check dataset
required_columns = [
    "brand", "ram", "storage",
    "battery", "camera", "screen", "price"
]

if not all(col in data.columns for col in required_columns):
    raise ValueError("Dataset columns are incorrect.")

data = data.dropna()

# Input features
X = data[
    ["brand", "ram", "storage",
     "battery", "camera", "screen"]
]

# Output variable
y = data["price"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "brand_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["brand"]
        )
    ],
    remainder="passthrough"
)

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Train model
print("Training ML model...")

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

# Display results
print("\nMODEL PERFORMANCE")
print("-------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# Save trained model
joblib.dump(
    model,
    BASE_DIR / "mobile_price_model.pkl"
)

# Save evaluation metrics
metrics = {
    "MAE": round(float(mae), 2),
    "RMSE": round(float(rmse), 2),
    "R2 Score": round(float(r2), 3)
}

with open(BASE_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel training completed!")
print("Model saved successfully!")
print("Metrics saved successfully!")