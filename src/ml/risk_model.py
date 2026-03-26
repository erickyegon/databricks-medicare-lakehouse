"""
src/ml/risk_model.py
─────────────────────
XGBoost risk stratification model with full MLflow integration:
  - Experiment tracking (params, metrics, artifacts)
  - SHAP explainability logged as artifacts
  - Calibration (isotonic regression) for accurate probability estimates
  - MLflow Model Registry with staging/production promotion
  - Drift monitoring via PSI across retraining cycles

Model targets:
  - risk_tier classification (low / moderate / high)  — XGBoost multiclass
  - annual cost regression                             — XGBoost regression
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from config.pipeline_config import CONFIG, MLConfig
from src.utils.spark_utils import get_logger, get_spark

logger = get_logger(__name__)


# ── Feature columns used by the model ─────────────────────────────────────

# ── Feature columns used by the model ─────────────────────────────────────
# Exactly 33 features — must match ml_features.risk_feature_store columns.
# Grouped to match the README data dictionary for auditability.

CLASSIFICATION_FEATURES = [
    # Demographic (5)
    "age", "sex_male", "dual_eligible", "demographic_raf", "age_bracket",
    # HCC burden (3)
    "max_hcc_burden", "max_hcc_raf", "max_interaction_raf",
    # Condition flags (8)
    "has_chf", "has_afib", "has_diabetes", "has_ckd",
    "has_cancer", "has_copd", "has_depression", "has_metastatic",
    # Pre-period utilization (11)
    "pre_ip_admits", "pre_ed_visits", "pre_specialist_visits",
    "pre_primary_visits", "pre_n_months", "pre_total_cost",
    "pre_avg_claim", "pre_max_claim", "pre_pmpm",
    "pre_ip_rate", "pre_ed_rate",
    # RAF and cost (3)
    "raf_score", "estimated_annual_cost", "actual_pmpm",
    # Log-transformed (3)
    "log_pre_pmpm", "log_estimated_cost", "log_pre_total_cost",
]

assert len(CLASSIFICATION_FEATURES) == 33, \
    f"Expected 33 features, got {len(CLASSIFICATION_FEATURES)}"

REGRESSION_FEATURES = CLASSIFICATION_FEATURES  # same feature set


# ── Model wrapper ──────────────────────────────────────────────────────────

class RiskStratificationModel:
    """
    Encapsulates the XGBoost classifier + regressor with calibration and SHAP.

    Attributes:
        clf:         Calibrated XGBoost classifier (risk_tier)
        reg:         XGBoost regressor (annual cost)
        label_enc:   LabelEncoder for risk_tier
        feature_cols:List of feature column names used
    """

    def __init__(self, config: Optional[MLConfig] = None):
        self.config      = config or CONFIG.ml
        self.clf         = None
        self.reg         = None
        self.label_enc   = LabelEncoder()
        self.feature_cols = CLASSIFICATION_FEATURES
        self._is_fitted  = False

    def _prep_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Split features and targets from the feature store DataFrame."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature store is missing {len(missing)} expected columns: {missing}\n"
                f"Check that Gold aggregation ran successfully and feature_cols "
                f"in risk_model.py matches the feature store schema."
            )
        X     = df[self.feature_cols].astype(float)
        y_clf = df[self.config.target_col]
        y_reg = df[self.config.cost_target_col]
        return X, y_clf, y_reg

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df:  Optional[pd.DataFrame] = None,
    ) -> "RiskStratificationModel":
        """Train classifier and regressor on the feature store data."""
        X_train, y_clf_train, y_reg_train = self._prep_data(train_df)

        # Encode labels
        y_enc = self.label_enc.fit_transform(y_clf_train)

        # ── Classifier ──────────────────────────────────────────────
        base_clf = xgb.XGBClassifier(
            **{k: v for k, v in self.config.xgb_params.items()
               if k not in ["use_label_encoder", "eval_metric"]},
            objective="multi:softprob",
            num_class=len(self.label_enc.classes_),
        )
        # Calibrate for reliable probability estimates (Platt / isotonic)
        self.clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        self.clf.fit(X_train, y_enc)

        # ── Regressor ───────────────────────────────────────────────
        self.reg = xgb.XGBRegressor(
            **{k: v for k, v in self.config.xgb_params.items()
               if k not in ["use_label_encoder", "eval_metric"]},
            objective="reg:squarederror",
        )
        self.reg.fit(X_train, y_reg_train)

        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions as a DataFrame with tier and cost columns."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        X = df[self.feature_cols].astype(float)
        tier_enc   = self.clf.predict(X)
        tier_proba = self.clf.predict_proba(X)
        tier_labels= self.label_enc.inverse_transform(tier_enc)
        cost_pred  = self.reg.predict(X)

        result = pd.DataFrame({
            "predicted_tier": tier_labels,
            "predicted_cost": cost_pred,
        })
        for i, cls in enumerate(self.label_enc.classes_):
            result[f"prob_{cls}"] = tier_proba[:, i]

        return result

    def evaluate(
        self, test_df: pd.DataFrame
    ) -> Dict:
        """Compute classification and regression metrics on a test set."""
        X_test, y_clf_test, y_reg_test = self._prep_data(test_df)
        preds = self.predict(test_df)

        # Classification metrics
        accuracy  = accuracy_score(y_clf_test, preds["predicted_tier"])
        clf_report= classification_report(
            y_clf_test, preds["predicted_tier"], output_dict=True
        )
        # AUC (one-vs-rest, macro)
        y_enc  = self.label_enc.transform(y_clf_test)
        proba_cols = [f"prob_{c}" for c in self.label_enc.classes_]
        try:
            auc = roc_auc_score(
                y_enc, preds[proba_cols].values,
                multi_class="ovr", average="macro"
            )
        except Exception:
            auc = None

        # Regression metrics
        mae = mean_absolute_error(y_reg_test, preds["predicted_cost"])
        r2  = r2_score(y_reg_test,            preds["predicted_cost"])

        return {
            "tier_accuracy":       round(accuracy, 4),
            "auc_macro":           round(auc, 4) if auc else None,
            "cost_mae":            round(float(mae), 2),
            "cost_r2":             round(float(r2), 4),
            "classification_report": clf_report,
        }


# ── SHAP explainability ────────────────────────────────────────────────────

def compute_shap_values(
    model: RiskStratificationModel,
    X:     pd.DataFrame,
    output_dir: str = "/tmp/shap",
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for the classifier base model.
    Saves a beeswarm plot and returns the shap values array.
    """
    try:
        import shap
        import matplotlib.pyplot as plt

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract the base XGBoost model from CalibratedClassifierCV
        base_model = model.clf.calibrated_classifiers_[0].estimator
        explainer  = shap.TreeExplainer(base_model)
        shap_vals  = explainer.shap_values(X.astype(float))

        # Beeswarm (mean |SHAP| across classes)
        if isinstance(shap_vals, list):
            mean_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            mean_shap = np.abs(shap_vals)

        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance = pd.Series(
            mean_shap.mean(axis=0), index=model.feature_cols
        ).sort_values(ascending=False).head(20)
        feature_importance.plot(kind="barh", ax=ax, color="#1B4F72")
        ax.set_title("SHAP Feature Importance (mean |SHAP| across risk classes)")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP plot saved: {output_dir}/shap_importance.png")
        return shap_vals

    except ImportError:
        logger.warning("SHAP not available — skipping SHAP computation")
        return None


# ── MLflow training run ────────────────────────────────────────────────────

def train_and_log(
    spark=None,
    config: Optional[MLConfig] = None,
    run_name: Optional[str] = None,
    register: bool = True,
) -> Tuple[RiskStratificationModel, Dict]:
    """
    Full training run with MLflow experiment tracking.

    Steps:
      1. Load feature store from Gold
      2. Train/test split
      3. Fit RiskStratificationModel
      4. Evaluate metrics
      5. Log params, metrics, model, SHAP artifacts to MLflow
      6. Register model in MLflow Model Registry
      7. Promote to Staging

    Returns:
        (fitted_model, metrics_dict)
    """
    spark  = spark or get_spark()
    config = config or CONFIG.ml

    mlflow.set_experiment(config.experiment_name)

    run_name = run_name or f"raf_risk_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"

    logger.info(f"Loading feature store: {config.feature_table}")
    feat_df = spark.table(config.feature_table).toPandas()
    feat_df = feat_df.fillna(0)

    # Ensure bool → int for XGBoost
    bool_cols = feat_df.select_dtypes("bool").columns
    feat_df[bool_cols] = feat_df[bool_cols].astype(int)

    logger.info(f"Feature store loaded: {len(feat_df):,} members, {len(feat_df.columns)} columns")

    X_train_df, X_test_df = train_test_split(
        feat_df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=feat_df[config.target_col],
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # ── Log parameters ────────────────────────────────────────
        mlflow.log_params({
            "n_train":         len(X_train_df),
            "n_test":          len(X_test_df),
            "n_features":      len(CLASSIFICATION_FEATURES),
            "test_size":       config.test_size,
            "random_state":    config.random_state,
            "calibration":     "isotonic",
            **{f"xgb_{k}": v for k, v in config.xgb_params.items()
               if k not in ["use_label_encoder"]},
        })

        # ── Train ─────────────────────────────────────────────────
        model = RiskStratificationModel(config)
        model.fit(X_train_df)

        # ── Evaluate ──────────────────────────────────────────────
        metrics = model.evaluate(X_test_df)
        logger.info(f"Metrics: {metrics}")

        # Log scalar metrics
        mlflow.log_metrics({
            "tier_accuracy": metrics["tier_accuracy"],
            "cost_mae":      metrics["cost_mae"],
            "cost_r2":       metrics["cost_r2"],
            **({"auc_macro": metrics["auc_macro"]} if metrics["auc_macro"] else {}),
        })

        # ── SHAP ─────────────────────────────────────────────────
        X_test_sample = X_test_df[CLASSIFICATION_FEATURES].head(200).astype(float)
        shap_vals = compute_shap_values(model, X_test_sample)
        if shap_vals is not None:
            mlflow.log_artifact("/tmp/shap/shap_importance.png")

        # ── Log classification report ──────────────────────────
        report_path = "/tmp/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(metrics["classification_report"], f, indent=2)
        mlflow.log_artifact(report_path)

        # ── Log feature list ───────────────────────────────────
        feat_path = "/tmp/feature_list.txt"
        with open(feat_path, "w") as f:
            f.write("\n".join(CLASSIFICATION_FEATURES))
        mlflow.log_artifact(feat_path)

        # ── Log model ──────────────────────────────────────────
        # Log underlying XGBoost base model for native xgb inference
        base_clf = model.clf.calibrated_classifiers_[0].estimator
        mlflow.xgboost.log_model(
            base_clf,
            artifact_path="xgboost_classifier",
            registered_model_name=config.model_name if register else None,
        )

        # Also log full sklearn pipeline (calibrated) via mlflow.sklearn
        import mlflow.sklearn
        mlflow.sklearn.log_model(
            model.clf,
            artifact_path="calibrated_classifier",
        )
        mlflow.sklearn.log_model(
            model.reg,
            artifact_path="cost_regressor",
        )

        # ── Promote to Staging ─────────────────────────────────
        if register:
            client = MlflowClient()
            try:
                latest = client.get_latest_versions(config.model_name, stages=["None"])
                if latest:
                    version = latest[-1].version
                    client.transition_model_version_stage(
                        name=config.model_name,
                        version=version,
                        stage=config.staging_alias,
                    )
                    logger.info(f"Model v{version} promoted to {config.staging_alias}")
            except Exception as e:
                logger.warning(f"Registry promotion skipped: {e}")

        logger.info(f"MLflow run complete: {run_id}")

    return model, metrics


# ── Drift monitoring — see src/ml/drift_monitor.py ───────────────────────
# PSI computation and feature drift monitoring have been extracted to a
# dedicated module for independent testability and separation of concerns.
# Import from there directly:
#
#   from src.ml.drift_monitor import monitor_drift, compute_psi, PSIReport
