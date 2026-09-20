import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Find the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File locations
DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "flood_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "flood_model.pkl"
)

# Load flood training data
data = pd.read_csv(DATA_FILE)

# Input features
X = data[
    [
        "rainfall",
        "water_level",
        "river_level",
        "soil_moisture",
        "humidity"
    ]
]

# Target value
y = data["flood_risk"]

# Create AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

# Create model folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Save trained model
joblib.dump(model, MODEL_FILE)

print("====================================")
print("AI Flood Risk Model")
print("====================================")
print("Training completed successfully!")
print("Model saved to:")
print(MODEL_FILE)
