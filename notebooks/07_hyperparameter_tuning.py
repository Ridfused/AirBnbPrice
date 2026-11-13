
"""
AirbnbPrice — Step 07: Hyperparameter Tuning & Model Optimization

Run from project root:
    python -u notebooks/07_hyperparameter_tuning.py

Input:
    data/processed/airbnb_india_training.csv

Target:
    price
"""

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint, loguniform, uniform

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "airbnb_india_training.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "price"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER = 25


# ============================================================
# 2. PREPROCESSING
# ============================================================

def make_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        c for c in X.columns
        if c not in numeric_columns
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            sparse=False
        )

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder)
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_columns),
            ("categorical", categorical_pipe, categorical_columns)
        ],
        remainder="drop",
        sparse_threshold=0.0
    )


# ============================================================
# 3. TARGET TRANSFORMATION
# ============================================================

def wrap_estimator(model, X):

    pipe = Pipeline([
        ("preprocess", make_preprocessor(X)),
        ("model", model)
    ])

    return TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False
    )


# ============================================================
# 4. CANDIDATE MODELS
# ============================================================

def candidate_models():

    candidates = [

        (
            "RandomForest",
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
            {
                "regressor__model__n_estimators":
                    randint(250, 901),

                "regressor__model__max_depth":
                    [None, 5, 8, 12, 16, 22],

                "regressor__model__min_samples_split":
                    randint(2, 12),

                "regressor__model__min_samples_leaf":
                    randint(1, 7),

                "regressor__model__max_features":
                    [1.0, "sqrt", 0.7]
            }
        ),

        (
            "ExtraTrees",
            ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
            {
                "regressor__model__n_estimators":
                    randint(250, 901),

                "regressor__model__max_depth":
                    [None, 5, 8, 12, 16, 22],

                "regressor__model__min_samples_split":
                    randint(2, 12),

                "regressor__model__min_samples_leaf":
                    randint(1, 7),

                "regressor__model__max_features":
                    [1.0, "sqrt", 0.7]
            }
        ),

        (
            "HistGradientBoosting",
            HistGradientBoostingRegressor(
                random_state=RANDOM_STATE
            ),
            {
                "regressor__model__learning_rate":
                    loguniform(0.015, 0.2),

                "regressor__model__max_iter":
                    randint(100, 501),

                "regressor__model__max_leaf_nodes":
                    randint(7, 32),

                "regressor__model__l2_regularization":
                    loguniform(1e-4, 10),

                "regressor__model__min_samples_leaf":
                    randint(5, 31)
            }
        )
    ]

    # XGBoost (optional)

    try:
        from xgboost import XGBRegressor

        candidates.append(
            (
                "XGBoost",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist",
                    verbosity=0
                ),
                {
                    "regressor__model__n_estimators":
                        randint(200, 801),

                    "regressor__model__max_depth":
                        randint(2, 9),

                    "regressor__model__learning_rate":
                        loguniform(0.01, 0.2),

                    "regressor__model__subsample":
                        uniform(0.65, 0.35),

                    "regressor__model__colsample_bytree":
                        uniform(0.65, 0.35),

                    "regressor__model__reg_lambda":
                        loguniform(0.1, 20)
                }
            )
        )

    except Exception as exc:
        print(f"[INFO] XGBoost skipped: {exc}")

    # CatBoost (optional)

    try:
        from catboost import CatBoostRegressor

        candidates.append(
            (
                "CatBoost",
                CatBoostRegressor(
                    loss_function="RMSE",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1
                ),
                {
                    "regressor__model__iterations":
                        randint(250, 901),

                    "regressor__model__depth":
                        randint(4, 9),

                    "regressor__model__learning_rate":
                        loguniform(0.01, 0.2),

                    "regressor__model__l2_leaf_reg":
                        loguniform(1, 15)
                }
            )
        )

    except Exception as exc:
        print(f"[INFO] CatBoost skipped: {exc}")

    return candidates


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def metrics(y_true, y_pred):

    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),

        "R2": float(
            r2_score(y_true, y_pred)
        )
    }


# ============================================================
# 6. MAIN TRAINING FUNCTION
# ============================================================

def main():

    start_all = time.time()

    print("=" * 72)
    print("AIRBNBPRICE — STEP 07")
    print("HYPERPARAMETER TUNING & MODEL OPTIMIZATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(
        subset=[TARGET]
    )

    df = df[
        df[TARGET] > 0
    ].copy()

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} usable rows."
        )

    # Remove identifiers and listing text.

    drop_cols = [
        c for c in [
            "name",
            "address",
            "listing_id",
            "id"
        ]
        if c in df.columns
    ]

    X = df.drop(
        columns=[TARGET] + drop_cols
    )

    y = df[TARGET].astype(float)

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns found."
        )

    print(f"\nRows: {len(df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Columns: {X.columns.tolist()}")

    print(f"\nPrice median: {y.median():,.2f}")
    print(f"Price mean: {y.mean():,.2f}")
    print(f"Price minimum: {y.min():,.2f}")
    print(f"Price maximum: {y.max():,.2f}")

    print(
        "\nPrice units are inherited from the source dataset."
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # --------------------------------------------------------
    # Baseline model
    # --------------------------------------------------------

    print("\nTraining median baseline...")

    baseline = DummyRegressor(
        strategy="median"
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_pred = baseline.predict(
        X_test
    )

    baseline_result = {
        "Model": "MedianBaseline",
        **metrics(y_test, baseline_pred)
    }

    print("\nBaseline results:")
    print(baseline_result)

    # --------------------------------------------------------
    # Cross-validation setup
    # --------------------------------------------------------

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_results = []
    fitted_searches = []

    # --------------------------------------------------------
    # Hyperparameter tuning
    # --------------------------------------------------------

    models = candidate_models()

    print("\nModels selected for tuning:")

    for name, _, _ in models:
        print("-", name)

    for name, estimator, distributions in models:

        print("\n" + "=" * 72)
        print(f"TUNING MODEL: {name}")
        print("=" * 72)

        wrapped = wrap_estimator(
            estimator,
            X_train
        )

        search = RandomizedSearchCV(
            estimator=wrapped,
            param_distributions=distributions,
            n_iter=N_ITER,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=2,
            error_score="raise",
            return_train_score=False
        )

        t0 = time.time()

        print(
            f"\nStarting {N_ITER} candidates "
            f"with {CV_FOLDS}-fold cross-validation..."
        )

        search.fit(
            X_train,
            y_train
        )

        elapsed = time.time() - t0

        predictions = search.best_estimator_.predict(
            X_test
        )

        result = {
            "Model": name,
            **metrics(y_test, predictions),
            "CV_MAE": float(-search.best_score_),
            "Fit_Seconds": round(elapsed, 2),
            "Best_Params": json.dumps(
                search.best_params_,
                default=str
            )
        }

        all_results.append(result)

        fitted_searches.append(
            (
                name,
                search.best_estimator_,
                search.best_score_
            )
        )

        print("\nBest parameters:")
        print(search.best_params_)

        print("\nModel results:")
        print(result)

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    ).sort_values(
        "CV_MAE",
        ascending=True
    )

    results_df.to_csv(
        REPORT_DIR / "tuning_comparison.csv",
        index=False
    )

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)

    display_cols = [
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "CV_MAE",
        "Fit_Seconds"
    ]

    print(
        results_df[display_cols].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_name = results_df.iloc[0]["Model"]

    best_model = next(
        model
        for name, model, _ in fitted_searches
        if name == best_name
    )

    best_pred = best_model.predict(
        X_test
    )

    best_test_metrics = metrics(
        y_test,
        best_pred
    )

    print("\nBEST MODEL:", best_name)
    print("Test metrics:", best_test_metrics)

    # --------------------------------------------------------
    # Geographic validation
    # --------------------------------------------------------

    geo_result = None

    city_col = (
        "city"
        if "city" in X.columns
        else None
    )

    if city_col and X[city_col].nunique() >= 5:

        groups = X[city_col].fillna(
            "UNKNOWN"
        ).astype(str)

        group_cv = GroupKFold(
            n_splits=min(
                5,
                groups.nunique()
            )
        )

        geo_scores = []

        print(
            "\nRunning geographic GroupKFold validation..."
        )

        for fold, (tr_idx, va_idx) in enumerate(
            group_cv.split(X, y, groups),
            start=1
        ):

            fold_model = clone(best_model)

            fold_model.fit(
                X.iloc[tr_idx],
                y.iloc[tr_idx]
            )

            fold_pred = fold_model.predict(
                X.iloc[va_idx]
            )

            fold_metrics = metrics(
                y.iloc[va_idx],
                fold_pred
            )

            fold_metrics["Fold"] = fold

            fold_metrics["HeldOutCities"] = int(
                groups.iloc[va_idx].nunique()
            )

            geo_scores.append(
                fold_metrics
            )

            print(
                f"Geo fold {fold}: "
                f"MAE={fold_metrics['MAE']:.2f}, "
                f"R2={fold_metrics['R2']:.3f}"
            )

        geo_df = pd.DataFrame(
            geo_scores
        )

        geo_df.to_csv(
            REPORT_DIR / "geographic_cv_results.csv",
            index=False
        )

        geo_result = {
            "mean_MAE": float(
                geo_df["MAE"].mean()
            ),

            "mean_RMSE": float(
                geo_df["RMSE"].mean()
            ),

            "mean_R2": float(
                geo_df["R2"].mean()
            )
        }

    else:

        print(
            "\n[INFO] Geographic CV skipped: "
            "city column unavailable or fewer than 5 cities."
        )

    # --------------------------------------------------------
    # Permutation feature importance
    # --------------------------------------------------------

    print(
        "\nCalculating permutation feature importance..."
    )

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance = pd.DataFrame({

        "feature": X.columns,

        "importance_mean":
            perm.importances_mean,

        "importance_std":
            perm.importances_std

    }).sort_values(
        "importance_mean",
        ascending=False
    )

    importance.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False
    )

    print("\nTop 10 important features:")

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save tuned model
    # --------------------------------------------------------

    tuned_model_path = (
        MODEL_DIR /
        "best_airbnb_price_model_tuned.joblib"
    )

    joblib.dump(
        best_model,
        tuned_model_path
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_report = X_test.copy()

    prediction_report["actual_price"] = y_test

    prediction_report["predicted_price"] = best_pred

    prediction_report["absolute_error"] = np.abs(
        y_test.to_numpy() - best_pred
    )

    prediction_report.to_csv(
        REPORT_DIR / "tuned_model_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {

        "best_model": best_name,

        "best_model_path":
            str(tuned_model_path.relative_to(ROOT)),

        "rows_used": int(len(df)),

        "features_used": X.columns.tolist(),

        "random_holdout_test_metrics":
            best_test_metrics,

        "geographic_group_cv":
            geo_result,

        "baseline_metrics":
            metrics(y_test, baseline_pred),

        "total_runtime_seconds":
            round(time.time() - start_all, 2)
    }

    with open(
        REPORT_DIR / "tuning_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 07 COMPLETE")
    print("=" * 72)

    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    print("\nSaved reports:")

    print(" - reports/tuning_comparison.csv")
    print(" - reports/geographic_cv_results.csv")
    print(" - reports/feature_importance.csv")
    print(" - reports/tuned_model_test_predictions.csv")
    print(" - reports/tuning_summary.json")

    print(
        "\nExisting baseline model was NOT overwritten."
    )

    print(
        "Tuned model saved to:",
        tuned_model_path
    )


# ============================================================
# 7. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
