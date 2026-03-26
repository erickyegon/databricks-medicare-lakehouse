# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04 — MLflow Risk Model Training & Registry
# MAGIC
# MAGIC **Purpose**: Train XGBoost risk stratification model against the Gold feature store,
# MAGIC track experiments in MLflow, and register the production model.
# MAGIC
# MAGIC ## Model outputs
# MAGIC - **Risk tier classifier**: low / moderate / high (XGBoost multiclass + isotonic calibration)
# MAGIC - **Annual cost regressor**: expected annual cost per member (XGBoost regression)
# MAGIC - **SHAP explainability**: per-feature contribution plots logged as MLflow artifacts
# MAGIC - **Drift monitoring**: PSI computed across key features
# MAGIC
# MAGIC **Expected runtime**: ~5–8 minutes

# COMMAND ----------

import sys, os
repo_root = os.path.abspath("..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md ## 0. Parameters

# COMMAND ----------

dbutils.widgets.text("run_name",   "", "MLflow run name (blank = auto)")
dbutils.widgets.dropdown("register_model", "yes", ["yes", "no"], "Register in Model Registry")

RUN_NAME = dbutils.widgets.get("run_name") or None
REGISTER = dbutils.widgets.get("register_model") == "yes"

print(f"Run name: {RUN_NAME or 'auto'} | Register: {REGISTER}")

# COMMAND ----------

# MAGIC %md ## 1. Load feature store

# COMMAND ----------

from config.pipeline_config import Tables, CONFIG

feat_df = spark.table(Tables.FEATURE_STORE)
print(f"Feature store: {feat_df.count():,} members | {len(feat_df.columns)} columns")

# Risk tier distribution
from pyspark.sql import functions as F
display(feat_df.groupBy("risk_tier").count().orderBy("risk_tier"))

# COMMAND ----------

# MAGIC %md ## 2. Train and log model

# COMMAND ----------

from src.ml.risk_model import train_and_log

model, metrics = train_and_log(
    spark=spark,
    run_name=RUN_NAME,
    register=REGISTER,
)

print("\n✅ Model training complete")
print(f"   Tier accuracy:  {metrics['tier_accuracy']:.1%}")
print(f"   AUC (macro):    {metrics['auc_macro']}")
print(f"   Cost MAE:       ${metrics['cost_mae']:,.2f}")
print(f"   Cost R²:        {metrics['cost_r2']:.4f}")

# COMMAND ----------

# MAGIC %md ## 3. Review MLflow experiment

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = mlflow.get_experiment_by_name(CONFIG.ml.experiment_name)
if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=5
    )
    import pandas as pd
    run_summary = pd.DataFrame([{
        "run_name":     r.data.tags.get("mlflow.runName", ""),
        "tier_accuracy":r.data.metrics.get("tier_accuracy", None),
        "cost_mae":     r.data.metrics.get("cost_mae", None),
        "cost_r2":      r.data.metrics.get("cost_r2", None),
        "auc_macro":    r.data.metrics.get("auc_macro", None),
        "status":       r.info.status,
        "start_time":   pd.to_datetime(r.info.start_time, unit="ms"),
    } for r in runs])
    display(run_summary)

# COMMAND ----------

# MAGIC %md ## 4. Model evaluation on test set

# COMMAND ----------

from src.ml.risk_model import CLASSIFICATION_FEATURES
import pandas as pd

feat_pd = spark.table(Tables.FEATURE_STORE).toPandas().fillna(0)
bool_cols = feat_pd.select_dtypes("bool").columns
feat_pd[bool_cols] = feat_pd[bool_cols].astype(int)

from sklearn.model_selection import train_test_split
_, test_df = train_test_split(
    feat_pd,
    test_size=0.20,
    random_state=42,
    stratify=feat_pd["risk_tier"]
)

preds = model.predict(test_df)
test_df = test_df.reset_index(drop=True)
results = pd.concat([
    test_df[["bene_id", "risk_tier", "raf_score", "estimated_annual_cost"]],
    preds
], axis=1)
results.columns = ["bene_id", "actual_tier", "raf_score", "actual_cost",
                   "predicted_tier", "predicted_cost", "prob_high", "prob_low", "prob_moderate"]
display(results.head(20))

# COMMAND ----------

# MAGIC %md ### Confusion matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(results["actual_tier"], results["predicted_tier"],
                      labels=["low", "moderate", "high"])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Moderate", "High"],
            yticklabels=["Low", "Moderate", "High"], ax=ax)
ax.set_xlabel("Predicted Tier")
ax.set_ylabel("Actual Tier")
ax.set_title(f"Confusion Matrix — Tier Accuracy: {metrics['tier_accuracy']:.1%}")
plt.tight_layout()
display(fig)

# COMMAND ----------

print(classification_report(results["actual_tier"], results["predicted_tier"]))

# COMMAND ----------

# MAGIC %md ### Predicted vs actual cost

# COMMAND ----------

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(results["actual_cost"], results["predicted_cost"],
           alpha=0.3, s=10, c="#1B4F72")
lim = [results["actual_cost"].min(), results["actual_cost"].max()]
ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Annual Cost ($)")
ax.set_ylabel("Predicted Annual Cost ($)")
ax.set_title(f"Cost Model — R²={metrics['cost_r2']:.3f}  MAE=${metrics['cost_mae']:,.0f}")
ax.legend()
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md ## 5. Drift monitoring — PSI

# COMMAND ----------

from src.ml.drift_monitor import monitor_drift, DEFAULT_DRIFT_FEATURES

# For demonstration: compare feature store to itself (in production,
# compare baseline cohort vs latest month's feature store)
report = monitor_drift(
    baseline_table       = Tables.FEATURE_STORE,
    current_table        = Tables.FEATURE_STORE,
    features_to_monitor  = DEFAULT_DRIFT_FEATURES,
    spark                = spark,
)

print(report.summary())
display(report.to_dataframe())

# COMMAND ----------

# MAGIC %md ## 6. Model Registry — check versions

# COMMAND ----------

if REGISTER:
    try:
        versions = client.search_model_versions(f"name='{CONFIG.ml.model_name}'")
        reg_summary = pd.DataFrame([{
            "version":     v.version,
            "stage":       v.current_stage,
            "run_id":      v.run_id[:8] + "...",
            "status":      v.status,
            "created":     pd.to_datetime(v.creation_timestamp, unit="ms"),
        } for v in versions])
        display(reg_summary)
    except Exception as e:
        print(f"Registry check: {e}")

# COMMAND ----------

# MAGIC %md ## ✅ MLflow run complete
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Tier accuracy | See above |
# MAGIC | AUC (macro OvR) | See above |
# MAGIC | Cost MAE | See above |
# MAGIC | Cost R² | See above |
# MAGIC | Model registered | `medicare_raf_risk_model` in Staging |
# MAGIC
# MAGIC **Next**: Run `05_governance_and_workflows` to set up Unity Catalog access controls
# MAGIC and schedule the full Bronze→Silver→Gold→Retrain pipeline as a Databricks Workflow.
