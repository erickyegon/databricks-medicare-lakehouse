# Setup Guide

## Step 1 — Clone repo into Databricks
Repos → Add Repo → https://github.com/erickyegon/databricks-medicare-lakehouse

## Step 2 — Install dependencies on cluster
```python
%pip install xgboost shap scikit-learn mlflow delta-spark
```

## Step 3 — Run notebooks in order
| Notebook | Purpose | Runtime |
|----------|---------|---------|
| 01_bronze_ingestion | Generate data + ingest to Delta | ~3 min |
| 02_silver_processing | Clean, dedup, HCC-map | ~5 min |
| 03_gold_aggregates | RAF scores, feature store | ~5 min |
| 04_mlflow_model | Train XGBoost, log to MLflow | ~5 min |
| 05_governance_and_workflows | Unity Catalog + Workflow setup | ~2 min |

## Step 4 — Import the Workflow
1. Open Databricks UI → Workflows → Create Job
2. Click "Edit JSON"
3. Paste contents of workflows/pipeline_workflow.json
4. Replace REPLACE_WITH_YOUR_CLUSTER_ID with your cluster ID
5. Save and activate

## Community Edition note
Unity Catalog GRANT/REVOKE statements require a Unity Catalog-enabled 
workspace. On Community Edition these are skipped gracefully — all 
other pipeline steps run normally.
