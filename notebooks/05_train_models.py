
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# =====================================
# 1. LOAD DATA
# =====================================

DATA_PATH = Path("data/processed/airbnb_india_training.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("AIRBNB INDIA - MACHINE LEARNING TRAINING")
print("=" * 55)

# =====================================
# 2. DEFINE FEATURES AND TARGET
# =====================================

X = df.drop(columns=["price"])
y = df["price"]

categorical_features = [
    "city",
    "state",
    "room_type"
]

numeric_features = [
    "latitude",
    "longitude",
    "max_guests",
    "stars",
    "is_superhost",
    "has_rating"
]

# =====================================
# 3. TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# =====================================
# 4. PREPROCESSING
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numeric_features
        )
    ]
)

# =====================================
# 5. DEFINE MODELS
# =====================================

models = {
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "XGBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "CatBoost": CatBoostRegressor(
        iterations=500,
        depth=5,
        learning_rate=0.03,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
        allow_writing_files=False
    )
}

# =====================================
# 6. TRAIN AND EVALUATE
# =====================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    if name == "CatBoost":
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_features
        )

        predictions = model.predict(X_test)

    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"{name} completed!")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

# =====================================
# 7. COMPARE MODELS
# =====================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE"
).reset_index(drop=True)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

# =====================================
# 8. SAVE BEST MODEL
# =====================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

model_path = MODEL_DIR / "best_airbnb_price_model.joblib"

joblib.dump(best_model, model_path)

print("\nBEST MODEL:", best_model_name)
print("Saved model:", model_path)

# Save evaluation report
results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nMODEL TRAINING COMPLETED!")
