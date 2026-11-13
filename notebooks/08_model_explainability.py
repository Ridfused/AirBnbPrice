
"""
AirbnbPrice — Step 08: Model Explainability
--------------------------------------------
Explains the tuned Airbnb price prediction model using:

1. Permutation Feature Importance
2. SHAP Summary Plot
3. SHAP Global Feature Importance
4. SHAP Individual Prediction Explanation

Run from project root:
    python -u notebooks/08_model_explainability.py

Uses:
    data/processed/airbnb_india_training.csv
    models/best_airbnb_price_model_tuned.joblib

The existing trained model is never overwritten.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from scipy import sparse

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor


warnings.filterwarnings("ignore")


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "airbnb_india_training.csv"

MODEL_FILE = ROOT / "models" / "best_airbnb_price_model_tuned.joblib"

REPORT_DIR = ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "price"

RANDOM_STATE = 42

TEST_SIZE = 0.20

SHAP_BACKGROUND_SIZE = 50

SHAP_EXPLAIN_SIZE = 50

MAX_DISPLAY = 20


# Create output directories.

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_data():

    print("\nLoading Airbnb dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(subset=[TARGET])

    df = df[df[TARGET] > 0].copy()

    # Remove identifiers and leakage-prone text fields.
    drop_columns = [
        col for col in [
            "name",
            "address",
            "listing_id",
            "id"
        ]
        if col in df.columns
    ]

    X = df.drop(
        columns=[TARGET] + drop_columns
    )

    y = df[TARGET].astype(float)

    if X.empty:
        raise ValueError("No usable feature columns found.")

    print("Dataset loaded successfully!")

    print("Rows:", len(df))

    print("Features:", X.shape[1])

    print("Feature columns:", X.columns.tolist())

    return X, y


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("\nLoading tuned model...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Tuned model not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print("Tuned model loaded successfully!")

    print("Model path:", MODEL_FILE)

    return model


# ============================================================
# 5. PREPARE TRAINING AND TEST DATA
# ============================================================

def prepare_data(X, y):

    print("\nPreparing training and test datasets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("Training rows:", len(X_train))

    print("Testing rows:", len(X_test))

    return X_train, X_test, y_train, y_test


# ============================================================
# 6. PERMUTATION FEATURE IMPORTANCE
# ============================================================

def calculate_permutation_importance(
    model,
    X_test,
    y_test
):

    print("\nCalculating permutation feature importance...")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({

        "feature": X_test.columns,

        "importance_mean": result.importances_mean,

        "importance_std": result.importances_std

    })

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    )

    importance_df.to_csv(
        REPORT_DIR / "permutation_feature_importance.csv",
        index=False
    )

    print("\nTOP PERMUTATION FEATURES")

    print(
        importance_df.head(20).to_string(index=False)
    )

    # Visualization.

    plot_data = importance_df.head(MAX_DISPLAY).sort_values(
        "importance_mean",
        ascending=True
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        plot_data["feature"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std"]
    )

    plt.xlabel("Decrease in model performance when shuffled")

    plt.ylabel("Feature")

    plt.title("Permutation Feature Importance")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "permutation_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Permutation importance completed!")

    return importance_df


# ============================================================
# 7. EXTRACT FITTED PREPROCESSOR AND ESTIMATOR
# ============================================================

def extract_pipeline(model):

    print("\nExtracting fitted model components...")

    if isinstance(model, TransformedTargetRegressor):

        fitted_pipeline = model.regressor_

    elif hasattr(model, "named_steps"):

        fitted_pipeline = model

    else:

        raise TypeError(
            "Unsupported model structure. "
            "Expected TransformedTargetRegressor or Pipeline."
        )

    if "preprocess" not in fitted_pipeline.named_steps:

        raise ValueError(
            "The fitted pipeline does not contain a "
            "'preprocess' step."
        )

    if "model" not in fitted_pipeline.named_steps:

        raise ValueError(
            "The fitted pipeline does not contain a "
            "'model' step."
        )

    preprocessor = fitted_pipeline.named_steps["preprocess"]

    estimator = fitted_pipeline.named_steps["model"]

    print("Preprocessor extracted.")

    print("Underlying estimator:", type(estimator).__name__)

    return preprocessor, estimator


# ============================================================
# 8. TRANSFORM FEATURES FOR SHAP
# ============================================================

def transform_features(
    preprocessor,
    X_train,
    X_test
):

    print("\nTransforming features for SHAP...")

    background = X_train.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(X_train)),
        random_state=RANDOM_STATE
    )

    explain_data = X_test.sample(
        n=min(SHAP_EXPLAIN_SIZE, len(X_test)),
        random_state=RANDOM_STATE
    )

    # IMPORTANT:
    # Transform categorical and numerical data using the
    # already-fitted preprocessing pipeline.
    #
    # This prevents SHAP from receiving raw string columns.

    background_transformed = preprocessor.transform(
        background
    )

    explain_transformed = preprocessor.transform(
        explain_data
    )

    # Convert sparse matrices into dense numerical arrays.

    if sparse.issparse(background_transformed):

        background_transformed = background_transformed.toarray()

    if sparse.issparse(explain_transformed):

        explain_transformed = explain_transformed.toarray()

    background_transformed = np.asarray(
        background_transformed,
        dtype=float
    )

    explain_transformed = np.asarray(
        explain_transformed,
        dtype=float
    )

    # Retrieve feature names after one-hot encoding.

    try:

        feature_names = list(
            preprocessor.get_feature_names_out()
        )

    except Exception:

        feature_names = [
            f"feature_{i}"
            for i in range(explain_transformed.shape[1])
        ]

    print("SHAP background shape:", background_transformed.shape)

    print("SHAP explanation shape:", explain_transformed.shape)

    print("Transformed feature count:", len(feature_names))

    return (
        background_transformed,
        explain_transformed,
        feature_names
    )


# ============================================================
# 9. SHAP EXPLAINABILITY ANALYSIS
# ============================================================

def calculate_shap_values(
    estimator,
    background_transformed,
    explain_transformed,
    feature_names
):

    print("\nInitializing SHAP Tree Explainer...")

    # The underlying estimator predicts log-transformed prices.
    # The outer TransformedTargetRegressor converts predictions
    # back to original price units.

    explainer = shap.TreeExplainer(
        estimator,
        data=background_transformed,
        feature_perturbation="interventional"
    )

    print("SHAP explainer initialized.")

    print("\nCalculating SHAP values...")

    shap_values = explainer(
        explain_transformed,
        check_additivity=False
    )

    explanation = shap.Explanation(

        values=shap_values.values,

        base_values=shap_values.base_values,

        data=explain_transformed,

        feature_names=feature_names

    )

    print("SHAP values calculated successfully!")

    return explanation


# ============================================================
# 10. SHAP SUMMARY PLOT
# ============================================================

def create_shap_summary_plot(explanation):

    print("\nGenerating SHAP summary plot...")

    shap.summary_plot(

        explanation.values,

        features=explanation.data,

        feature_names=explanation.feature_names,

        max_display=MAX_DISPLAY,

        show=False

    )

    plt.title(
        "SHAP Summary - Airbnb Price Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP summary plot saved.")


# ============================================================
# 11. SHAP GLOBAL FEATURE IMPORTANCE
# ============================================================

def create_shap_importance(explanation):

    print("\nCalculating SHAP global feature importance...")

    values = np.abs(
        explanation.values
    ).mean(axis=0)

    importance_df = pd.DataFrame({

        "feature": explanation.feature_names,

        "mean_absolute_shap": values

    })

    importance_df = importance_df.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    importance_df.to_csv(
        REPORT_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("\nTOP SHAP FEATURES")

    print(
        importance_df.head(20).to_string(index=False)
    )

    # SHAP bar visualization.

    shap.summary_plot(

        explanation.values,

        features=explanation.data,

        feature_names=explanation.feature_names,

        plot_type="bar",

        max_display=MAX_DISPLAY,

        show=False

    )

    plt.title(
        "SHAP Global Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP global importance saved.")

    return importance_df


# ============================================================
# 12. INDIVIDUAL PREDICTION EXPLANATION
# ============================================================

def explain_individual_prediction(explanation):

    print("\nGenerating individual prediction explanation...")

    if len(explanation) == 0:

        print("No predictions available for individual explanation.")

        return

    individual = explanation[0]

    shap.plots.waterfall(

        individual,

        max_display=MAX_DISPLAY,

        show=False

    )

    plt.title(
        "Individual Airbnb Price Prediction Explanation"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_individual_prediction.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Individual prediction explanation saved.")


# ============================================================
# 13. SAVE EXPLAINABILITY SUMMARY
# ============================================================

def save_summary(
    permutation_df,
    shap_importance_df,
    model
):

    print("\nSaving explainability summary...")

    summary = {

        "project": "AirbnbPrice",

        "step": "08 - Model Explainability",

        "model_type": type(model).__name__,

        "top_permutation_features":
            permutation_df.head(10)["feature"].tolist(),

        "top_shap_features":
            shap_importance_df.head(10)["feature"].tolist(),

        "figures": [

            "permutation_feature_importance.png",

            "shap_summary.png",

            "shap_feature_importance.png",

            "shap_individual_prediction.png"

        ],

        "reports": [

            "permutation_feature_importance.csv",

            "shap_feature_importance.csv"

        ],

        "note": (
            "SHAP values explain the underlying estimator's "
            "log-transformed target predictions."
        )

    }

    summary_file = REPORT_DIR / "explainability_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("Summary saved:", summary_file)


# ============================================================
# 14. MAIN EXECUTION
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 72)

    print("AIRBNBPRICE — STEP 08: MODEL EXPLAINABILITY")

    print("=" * 72)

    # Load dataset.

    X, y = load_data()

    # Load tuned model from Step 07.

    model = load_model()

    # Reproduce the same train/test split from Step 07.

    X_train, X_test, y_train, y_test = prepare_data(
        X,
        y
    )

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    permutation_df = calculate_permutation_importance(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # EXTRACT PREPROCESSOR AND ESTIMATOR
    # --------------------------------------------------------

    preprocessor, estimator = extract_pipeline(
        model
    )

    # --------------------------------------------------------
    # TRANSFORM DATA FOR SHAP
    # --------------------------------------------------------

    (
        background_transformed,
        explain_transformed,
        feature_names
    ) = transform_features(
        preprocessor,
        X_train,
        X_test
    )

    # --------------------------------------------------------
    # SHAP CALCULATION
    # --------------------------------------------------------

    explanation = calculate_shap_values(
        estimator,
        background_transformed,
        explain_transformed,
        feature_names
    )

    # --------------------------------------------------------
    # SHAP VISUALIZATIONS
    # --------------------------------------------------------

    create_shap_summary_plot(
        explanation
    )

    shap_importance_df = create_shap_importance(
        explanation
    )

    explain_individual_prediction(
        explanation
    )

    # --------------------------------------------------------
    # SAVE FINAL REPORT
    # --------------------------------------------------------

    save_summary(
        permutation_df,
        shap_importance_df,
        model
    )

    runtime = round(
        time.time() - start_time,
        2
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 08 COMPLETED SUCCESSFULLY!")

    print("=" * 72)

    print("\nMODEL:", type(estimator).__name__)

    print("RUNTIME:", runtime, "seconds")

    print("\nGENERATED FIGURES:")

    print("1. permutation_feature_importance.png")

    print("2. shap_summary.png")

    print("3. shap_feature_importance.png")

    print("4. shap_individual_prediction.png")

    print("\nGENERATED REPORTS:")

    print("1. permutation_feature_importance.csv")

    print("2. shap_feature_importance.csv")

    print("3. explainability_summary.json")

    print("\nAll outputs saved inside reports/.")

    print("\nExisting tuned model was NOT modified.")

    print("\nSTEP 08 FINISHED!")


# ============================================================
# 15. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
