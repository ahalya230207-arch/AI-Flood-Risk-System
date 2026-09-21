import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

print("====================================")
print("AI FLOOD RISK MODEL")
print("====================================")

print("Loading dataset...")

data = pd.read_csv(DATA_FILE)

print("Dataset rows:", len(data))

features = [
    "rainfall",
    "water_level",
    "river_level",
    "soil_moisture",
    "humidity"
]

X = data[features]

y = data["flood_risk"]

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X, y)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

print("")
print("Testing model...")
print("")

test_cases = [

    {
        "name": "LOW",
        "values": [5, 1.2, 1.7, 30, 50]
    },

    {
        "name": "MODERATE",
        "values": [35, 2.4, 2.9, 60, 74]
    },

    {
        "name": "HIGH",
        "values": [70, 3.4, 3.9, 82, 87]
    },

    {
        "name": "CRITICAL",
        "values": [130, 5.6, 6.2, 97, 97]
    }

]

for test in test_cases:

    prediction = model.predict(
        [test["values"]]
    )[0]

    prediction = max(
        0,
        min(
            100,
            prediction
        )
    )

    print(
        test["name"],
        "test ->",
        round(prediction, 2)
    )

print("")
print("====================================")
print("Training completed successfully!")
print("====================================")

print("Model saved to:")

print(MODEL_FILE)
