
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset and model
df = pd.read_csv("data/processed/airbnb_india_training.csv")

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = joblib.load(
    "models/best_airbnb_price_model.joblib"
)

predictions = model.predict(X_test)

# Actual vs predicted
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions,
})

comparison["Error"] = (
    comparison["Actual Price"] -
    comparison["Predicted Price"]
)

print("\nACTUAL VS PREDICTED")
print(comparison.head(15).to_string(index=False))

print("\nMODEL DIAGNOSTICS")
print("R2:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))

# Prediction scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions, alpha=0.7)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Airbnb Prices")

plt.tight_layout()
plt.savefig("reports/figures/actual_vs_predicted.png")
plt.show()

# Residual distribution
plt.figure(figsize=(8, 6))

plt.hist(comparison["Error"], bins=20)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")

plt.tight_layout()
plt.savefig("reports/figures/prediction_errors.png")
plt.show()

print("\nEvaluation completed!")
