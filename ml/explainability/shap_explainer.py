"""
SHAP (TreeSHAP) explainer for frequency & severity GBM models.

Provides SHAP-based feature attribution for catastrophe loss predictions,
explaining individual location risk factors and portfolio-level risk drivers.

Grounds all generated actuarial decisions in quantitative SHAP feature attributions
matching the design doc schema:
    explanations(prediction_id, location_id, feature_name, shap_value, rank)

Usage:
    python -m ml.explainability.shap_explainer [--num-samples N]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from ml.frequency_severity.build_training_table import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data_pipeline/bronze/explainability")
GOLD_FEATURES_PATH = Path("data_pipeline/gold/gold_features.parquet")
CLAIMS_PATH = Path("data_pipeline/silver/claims.parquet")


class SHAPExplainer:
    """
    TreeSHAP explainer for LightGBM/XGBoost catastrophe models.
    """

    def __init__(self, model: Any, feature_names: list[str], background_data: np.ndarray | None = None):
        """
        Initialize SHAP TreeExplainer.

        Args:
            model: Trained LightGBM or XGBoost model instance.
            feature_names: List of feature names corresponding to input matrix columns.
            background_data: Optional background sample array for SHAP calculation.
        """
        self.model = model
        self.feature_names = feature_names

        logger.info("Initializing TreeExplainer for model type %s...", type(model).__name__)
        if background_data is not None and len(background_data) > 100:
            # Subsample background data to speed up TreeSHAP initialization
            idx = np.random.choice(len(background_data), 100, replace=False)
            background_data = background_data[idx]

        if background_data is not None:
            self.explainer = shap.TreeExplainer(model, background_data)
        else:
            self.explainer = shap.TreeExplainer(model)

        logger.info("SHAP TreeExplainer initialized successfully.")

    def explain_instance(
        self,
        feature_vector: np.ndarray,
        location_id: str = "LOC_UNKNOWN",
        prediction_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute SHAP feature attribution for a single location prediction.

        Args:
            feature_vector: 1D or 2D array of feature values for 1 location.
            location_id: Identifier for the location.
            prediction_id: Unique prediction ID (generated if None).

        Returns:
            Dictionary containing prediction_id, location_id, predicted_value,
            top_drivers, and ranked feature attributions dataframe.
        """
        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector

        # Predict
        if hasattr(self.model, "predict_proba"):
            pred_val = float(self.model.predict_proba(X)[:, 1][0])
        elif hasattr(self.model, "predict"):
            pred_val = float(self.model.predict(X)[0])
        else:
            pred_val = 0.0

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            # For multi-class or binary classifier returning [class0_shap, class1_shap]
            vals = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif raw_shap.ndim == 3:
            vals = raw_shap[0, :, 1]
        elif raw_shap.ndim == 2:
            vals = raw_shap[0]
        else:
            vals = raw_shap

        if prediction_id is None:
            prediction_id = f"PRED_{location_id}_{int(np.random.randint(100000, 999999))}"

        # Build feature attribution records
        df = pd.DataFrame({
            "prediction_id": prediction_id,
            "location_id": location_id,
            "feature_name": self.feature_names,
            "feature_value": X[0],
            "shap_value": vals,
            "abs_shap": np.abs(vals),
        })

        # Rank features by absolute impact
        df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        top_drivers = df.head(5)[["feature_name", "shap_value", "feature_value"]].to_dict(orient="records")

        return {
            "prediction_id": prediction_id,
            "location_id": location_id,
            "predicted_value": pred_val,
            "top_drivers": top_drivers,
            "attributions": df.drop(columns=["abs_shap"]),
        }

    def explain_batch(
        self,
        X_matrix: np.ndarray,
        location_ids: list[str],
    ) -> pd.DataFrame:
        """
        Compute SHAP feature attributions for a batch of locations.

        Returns:
            DataFrame with columns matching explanations DB schema:
            [prediction_id, location_id, feature_name, feature_value, shap_value, rank]
        """
        raw_shap = self.explainer.shap_values(X_matrix)

        if isinstance(raw_shap, list):
            vals_matrix = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
        elif raw_shap.ndim == 3:
            vals_matrix = raw_shap[:, :, 1]
        else:
            vals_matrix = raw_shap

        records = []
        for idx, loc_id in enumerate(location_ids):
            pred_id = f"PRED_{loc_id}"
            feats = X_matrix[idx]
            shaps = vals_matrix[idx]

            sub_df = pd.DataFrame({
                "prediction_id": pred_id,
                "location_id": loc_id,
                "feature_name": self.feature_names,
                "feature_value": feats,
                "shap_value": shaps,
                "abs_shap": np.abs(shaps),
            })
            sub_df = sub_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
            sub_df["rank"] = sub_df.index + 1
            records.append(sub_df.drop(columns=["abs_shap"]))

        result_df = pd.concat(records, ignore_index=True)
        return result_df


def train_quick_explainable_model() -> tuple[Any, pd.DataFrame, list[str]]:
    """
    Helper function to fit an LightGBM severity model on gold features
    for SHAP explanation generation if pre-saved model binary is not found.
    """
    gold = pd.read_parquet(GOLD_FEATURES_PATH)
    claims = pd.read_parquet(CLAIMS_PATH)

    # Compute mean damage ratio per location as target
    loc_claims = claims.groupby("location_id")["damage_ratio"].mean().reset_index()
    data = gold.merge(loc_claims, on="location_id", how="left")
    data["damage_ratio"] = data["damage_ratio"].fillna(0.0)

    feature_cols = [
        "distance_to_coast_km",
        "tiv_usd",
        "year_built",
        "mslp_hpa_mean",
        "era5_wind_speed_ms_mean",
        "era5_wind_speed_ms_max",
    ]

    # Handle categorical encodings if present
    if "roof_type" in data.columns:
        roof_dummies = pd.get_dummies(data["roof_type"], prefix="roof", dtype=float)
        data = pd.concat([data, roof_dummies], axis=1)
        feature_cols.extend(roof_dummies.columns.tolist())

    if "construction_class" in data.columns:
        const_dummies = pd.get_dummies(data["construction_class"], prefix="construction", dtype=float)
        data = pd.concat([data, const_dummies], axis=1)
        feature_cols.extend(const_dummies.columns.tolist())

    X = data[feature_cols].fillna(0).values
    y = data["damage_ratio"].values

    model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X, y)

    return model, data, feature_cols


def run_shap_analysis(num_samples: int = 200) -> pd.DataFrame:
    """
    Run SHAP analysis pipeline and export explanations parquet table.
    """
    logger.info("Starting SHAP analysis pipeline (%d sample locations)...", num_samples)

    model, data, feature_cols = train_quick_explainable_model()

    sample_data = data.sample(n=min(num_samples, len(data)), random_state=42).reset_index(drop=True)
    X_sample = sample_data[feature_cols].fillna(0).values
    location_ids = sample_data["location_id"].tolist()

    explainer = SHAPExplainer(model, feature_cols, background_data=X_sample)
    explanations_df = explainer.explain_batch(X_sample, location_ids)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "shap_explanations.parquet"
    explanations_df.to_parquet(out_path, index=False)

    logger.info("SHAP explanations saved to %s (%d rows)", out_path, len(explanations_df))
    return explanations_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="SHAP Explainer Pipeline")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of locations to explain")
    args = parser.parse_args()

    df = run_shap_analysis(num_samples=args.num_samples)

    print("\n" + "=" * 60)
    print("SHAP EXPLANATIONS SAMPLE OUTPUT")
    print("=" * 60)
    print(df.head(15).to_string(index=False))
    print("=" * 60)
