import pandas as pd
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create simulated battery dataset
data = []

for i in range(1000):

    voltage = round(random.uniform(10.5, 13.0), 2)
    current = round(random.uniform(-2.0, 2.0), 2)
    temperature = round(random.uniform(20, 50), 1)
    soc = random.randint(5, 100)

    # Decide battery condition
    if voltage >= 12.2 and temperature < 35 and soc >= 60:
        health = "Healthy"

    elif voltage < 11.5 or temperature > 42 or soc < 20:
        health = "Critical"

    else:
        health = "Needs Attention"

    data.append([
        voltage,
        current,
        temperature,
        soc,
        health
    ])

# Create dataset
df = pd.DataFrame(
    data,
    columns=[
        "Voltage",
        "Current",
        "Temperature",
        "SOC",
        "Health"
    ]
)

# Save dataset
df.to_csv("battery_dataset.csv", index=False)

# Input features
X = df[
    [
        "Voltage",
        "Current",
        "Temperature",
        "SOC"
    ]
]

# Output/target
y = df["Health"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save trained model
joblib.dump(
    model,
    "battery_health_model.pkl"
)

print("Model saved successfully!")