
"""
AirbnbPrice — Step 09: Model Optimization & Final Validation

Tasks:
1. Baseline vs Tuned Model Comparison
2. Training vs Testing Performance
3. 5-Fold Cross Validation
4. Residual Analysis
5. Actual vs Predicted Visualization
6. Final Validation Report

Run:
python -u notebooks/09_final_validation.py
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

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "airbnb_india_training.csv"

BASELINE_MODEL = ROOT / "models" / "best_airbnb_price_model.joblib"

TUNED_MODEL = ROOT / "models" / "best_airbnb_price_model_tuned.joblib"

REPORT_DIR = ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "price"

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
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
            f"Target column '{TARGET}' is missing."
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(subset=[TARGET])

    df = df[df[TARGET] > 0].copy()

    # Keep the same feature exclusions used in Step 08.

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
        raise ValueError("No usable features found.")

    print("Dataset loaded successfully!")

    print("Rows:", len(df))

    print("Features:", X.shape[1])

    print("Target:", TARGET)

    return X, y


# ============================================================
# 4. LOAD TRAINED MODELS
# ============================================================

def load_models():

    print("\nLoading trained models...")

    models = {}

    if BASELINE_MODEL.exists():

        models["Baseline Random Forest"] = joblib.load(
            BASELINE_MODEL
        )

        print("Baseline model loaded.")

    else:

        print("[WARNING] Baseline model not found.")

    if TUNED_MODEL.exists():

        models["Tuned ExtraTrees"] = joblib.load(
            TUNED_MODEL
        )

        print("Tuned model loaded.")

    else:

        print("[WARNING] Tuned model not found.")

    if not models:

        raise FileNotFoundError(
            "No trained models found inside models/."
        )

    return models


# ============================================================
# 5. METRIC CALCULATION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)

    predicted = np.asarray(predicted)

    return {

        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),

        "R2": float(
            r2_score(actual, predicted)
        )

    }


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

def prepare_data(X, y):

    print("\nPreparing train/test split...")

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
# 7. BASELINE VS TUNED MODEL COMPARISON
# ============================================================

def compare_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 65)

    print("BASELINE VS TUNED MODEL COMPARISON")

    print("=" * 65)

    results = []

    predictions = {}

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        # Train and test predictions.

        train_pred = model.predict(X_train)

        test_pred = model.predict(X_test)

        train_metrics = calculate_metrics(
            y_train,
            train_pred
        )

        test_metrics = calculate_metrics(
            y_test,
            test_pred
        )

        predictions[name] = {

            "actual": y_test.to_numpy(),

            "predicted": np.asarray(test_pred),

            "train_predicted": np.asarray(train_pred)

        }

        results.append({

            "Model": name,

            "Train_MAE": train_metrics["MAE"],

            "Test_MAE": test_metrics["MAE"],

            "Train_RMSE": train_metrics["RMSE"],

            "Test_RMSE": test_metrics["RMSE"],

            "Train_R2": train_metrics["R2"],

            "Test_R2": test_metrics["R2"],

            "R2_Gap": (
                train_metrics["R2"] - test_metrics["R2"]
            )

        })

        print("Training metrics:", train_metrics)

        print("Testing metrics:", test_metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Test_MAE"
    )

    results_df.to_csv(
        REPORT_DIR / "final_model_comparison.csv",
        index=False
    )

    print("\nFINAL MODEL COMPARISON")

    print(
        results_df.to_string(index=False)
    )

    return results_df, predictions


# ============================================================
# 8. OVERFITTING ANALYSIS
# ============================================================

def analyze_overfitting(results_df):

    print("\n" + "=" * 65)

    print("OVERFITTING ANALYSIS")

    print("=" * 65)

    analysis = []

    for _, row in results_df.iterrows():

        gap = row["R2_Gap"]

        if gap < 0.05:

            status = "Low train-test R2 gap"

        elif gap < 0.15:

            status = "Moderate train-test R2 gap"

        else:

            status = "High train-test R2 gap"

        analysis.append({

            "Model": row["Model"],

            "Train_R2": row["Train_R2"],

            "Test_R2": row["Test_R2"],

            "R2_Gap": gap,

            "Assessment": status

        })

        print(f"\nModel: {row['Model']}")

        print(f"R2 Gap: {gap:.4f}")

        print("Assessment:", status)

    analysis_df = pd.DataFrame(analysis)

    analysis_df.to_csv(
        REPORT_DIR / "overfitting_analysis.csv",
        index=False
    )

    return analysis_df


# ============================================================
# 9. FIVE-FOLD CROSS VALIDATION
# ============================================================

def run_cross_validation(
    models,
    X,
    y
):

    print("\n" + "=" * 65)

    print("5-FOLD CROSS VALIDATION")

    print("=" * 65)

    cv = KFold(

        n_splits=CV_FOLDS,

        shuffle=True,

        random_state=RANDOM_STATE

    )

    scoring = {

        "MAE": "neg_mean_absolute_error",

        "MSE": "neg_mean_squared_error",

        "R2": "r2"

    }

    cv_results = []

    for name, model in models.items():

        print(f"\nRunning CV for {name}...")

        scores = cross_validate(

            model,

            X,

            y,

            cv=cv,

            scoring=scoring,

            n_jobs=-1,

            error_score="raise"

        )

        fold_mae = -scores["test_MAE"]

        fold_rmse = np.sqrt(
            -scores["test_MSE"]
        )

        fold_r2 = scores["test_R2"]

        result = {

            "Model": name,

            "Mean_MAE": float(np.mean(fold_mae)),

            "Std_MAE": float(np.std(fold_mae)),

            "Mean_RMSE": float(np.mean(fold_rmse)),

            "Std_RMSE": float(np.std(fold_rmse)),

            "Mean_R2": float(np.mean(fold_r2)),

            "Std_R2": float(np.std(fold_r2))

        }

        cv_results.append(result)

        print("Mean MAE:", result["Mean_MAE"])

        print("Mean RMSE:", result["Mean_RMSE"])

        print("Mean R2:", result["Mean_R2"])

    cv_df = pd.DataFrame(cv_results)

    cv_df.to_csv(
        REPORT_DIR / "cross_validation_results.csv",
        index=False
    )

    print("\nCROSS VALIDATION COMPLETED")

    print(cv_df.to_string(index=False))

    return cv_df


# ============================================================
# 10. ACTUAL VS PREDICTED VISUALIZATION
# ============================================================

def plot_actual_vs_predicted(
    predictions
):

    print("\nGenerating actual vs predicted plots...")

    for name, data in predictions.items():

        actual = data["actual"]

        predicted = data["predicted"]

        plt.figure(figsize=(9, 7))

        plt.scatter(
            actual,
            predicted,
            alpha=0.65
        )

        lower = min(
            np.min(actual),
            np.min(predicted)
        )

        upper = max(
            np.max(actual),
            np.max(predicted)
        )

        plt.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--"
        )

        plt.xlabel("Actual Price")

        plt.ylabel("Predicted Price")

        plt.title(
            f"Actual vs Predicted - {name}"
        )

        plt.tight_layout()

        filename = (
            name.lower().replace(" ", "_")
            + "_actual_vs_predicted.png"
        )

        plt.savefig(
            FIGURE_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    print("Actual vs predicted plots saved.")


# ============================================================
# 11. RESIDUAL ANALYSIS
# ============================================================

def plot_residuals(predictions):

    print("\nGenerating residual analysis...")

    residual_records = []

    for name, data in predictions.items():

        actual = data["actual"]

        predicted = data["predicted"]

        residuals = actual - predicted

        residual_records.append(

            pd.DataFrame({

                "Model": name,

                "Actual": actual,

                "Predicted": predicted,

                "Residual": residuals,

                "Absolute_Error": np.abs(residuals)

            })

        )

        # Residual vs predicted.

        plt.figure(figsize=(9, 6))

        plt.scatter(
            predicted,
            residuals,
            alpha=0.65
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.xlabel("Predicted Price")

        plt.ylabel("Residual")

        plt.title(
            f"Residual Analysis - {name}"
        )

        plt.tight_layout()

        filename = (
            name.lower().replace(" ", "_")
            + "_residuals.png"
        )

        plt.savefig(
            FIGURE_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # Residual distribution.

        plt.figure(figsize=(9, 6))

        plt.hist(
            residuals,
            bins=25
        )

        plt.xlabel("Prediction Residual")

        plt.ylabel("Frequency")

        plt.title(
            f"Residual Distribution - {name}"
        )

        plt.tight_layout()

        filename = (
            name.lower().replace(" ", "_")
            + "_residual_distribution.png"
        )

        plt.savefig(
            FIGURE_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    residual_df = pd.concat(
        residual_records,
        ignore_index=True
    )

    residual_df.to_csv(
        REPORT_DIR / "final_residual_analysis.csv",
        index=False
    )

    print("Residual analysis saved.")

    return residual_df


# ============================================================
# 12. FINAL MODEL SELECTION
# ============================================================

def select_final_model(results_df, cv_df):

    print("\n" + "=" * 65)

    print("FINAL MODEL SELECTION")

    print("=" * 65)

    # Step 07 already selected the tuned model.
    # We use holdout MAE and CV performance as validation.

    best_row = results_df.iloc[0]

    best_name = best_row["Model"]

    cv_row = cv_df[
        cv_df["Model"] == best_name
    ].iloc[0]

    final_result = {

        "selected_model": best_name,

        "test_MAE": float(best_row["Test_MAE"]),

        "test_RMSE": float(best_row["Test_RMSE"]),

        "test_R2": float(best_row["Test_R2"]),

        "train_R2": float(best_row["Train_R2"]),

        "R2_gap": float(best_row["R2_Gap"]),

        "cross_validation_MAE": float(
            cv_row["Mean_MAE"]
        ),

        "cross_validation_RMSE": float(
            cv_row["Mean_RMSE"]
        ),

        "cross_validation_R2": float(
            cv_row["Mean_R2"]
        )

    }

    print("\nSelected model:", best_name)

    print(
        json.dumps(
            final_result,
            indent=4
        )
    )

    return final_result


# ============================================================
# 13. SAVE FINAL VALIDATION REPORT
# ============================================================

def save_final_report(
    final_result,
    results_df,
    cv_df,
    overfitting_df,
    runtime
):

    report = {

        "project": "AirbnbPrice",

        "step": "09 - Model Optimization and Final Validation",

        "selected_model": final_result,

        "model_comparison": results_df.to_dict(
            orient="records"
        ),

        "cross_validation": cv_df.to_dict(
            orient="records"
        ),

        "overfitting_analysis": overfitting_df.to_dict(
            orient="records"
        ),

        "runtime_seconds": runtime,

        "model_files_modified": False

    }

    output_file = REPORT_DIR / "final_validation_summary.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print("\nFinal report saved:", output_file)


# ============================================================
# 14. MAIN FUNCTION
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)

    print("AIRBNBPRICE — STEP 09")

    print("MODEL OPTIMIZATION & FINAL VALIDATION")

    print("=" * 70)

    # Load dataset.

    X, y = load_data()

    # Load trained models.

    models = load_models()

    # Reproduce Step 07/08 holdout split.

    X_train, X_test, y_train, y_test = prepare_data(
        X,
        y
    )

    # Compare models.

    results_df, predictions = compare_models(

        models,

        X_train,

        X_test,

        y_train,

        y_test

    )

    # Overfitting analysis.

    overfitting_df = analyze_overfitting(
        results_df
    )

    # Five-fold cross validation.

    cv_df = run_cross_validation(
        models,
        X,
        y
    )

    # Visualizations.

    plot_actual_vs_predicted(
        predictions
    )

    residual_df = plot_residuals(
        predictions
    )

    # Select final candidate.

    final_result = select_final_model(
        results_df,
        cv_df
    )

    # Save summary.

    runtime = round(
        time.time() - start_time,
        2
    )

    save_final_report(

        final_result,

        results_df,

        cv_df,

        overfitting_df,

        runtime

    )

    # Final output.

    print("\n" + "=" * 70)

    print("STEP 09 COMPLETED SUCCESSFULLY!")

    print("=" * 70)

    print("\nSelected model:", final_result["selected_model"])

    print("Test MAE:", final_result["test_MAE"])

    print("Test RMSE:", final_result["test_RMSE"])

    print("Test R2:", final_result["test_R2"])

    print("CV R2:", final_result["cross_validation_R2"])

    print("\nAll reports saved inside reports/.")

    print("All visualizations saved inside reports/figures/.")

    print("\nExisting trained models were NOT modified.")

    print("\nSTEP 09 FINISHED!")


# ============================================================
# 15. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
