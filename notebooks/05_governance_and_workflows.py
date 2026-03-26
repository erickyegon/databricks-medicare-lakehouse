# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05 — Unity Catalog Governance & Workflow Orchestration
# MAGIC
# MAGIC **Purpose**: Set up Unity Catalog access controls, data lineage tags,
# MAGIC and analyst-facing views. Then define the production Databricks Workflow
# MAGIC that orchestrates the full pipeline on a daily schedule.
# MAGIC
# MAGIC ## What this notebook covers
# MAGIC 1. Create Unity Catalog structure (catalog → schemas)
# MAGIC 2. Set GRANT/REVOKE access controls by role group
# MAGIC 3. Add TBLPROPERTIES for HIPAA tagging and lineage
# MAGIC 4. Create analyst-facing Gold views
# MAGIC 5. Show the Workflow JSON definition (import to Databricks Jobs UI)
# MAGIC 6. Delta OPTIMIZE and VACUUM maintenance

# COMMAND ----------

import sys, os
repo_root = os.path.abspath("..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md ## 1. Unity Catalog setup

# COMMAND ----------

from src.governance.unity_catalog_setup import run_governance_setup

result = run_governance_setup(spark=spark)

print("✅ Unity Catalog governance configured")
print(f"   Catalog:  {result['catalog']}")
print(f"   Schemas:  {result['schemas']}")
print(f"   Groups:   {list(result['groups'].keys())}")
print(f"   Views:    {result['views_created']}")

# COMMAND ----------

# MAGIC %md ## 2. Verify catalog structure

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW SCHEMAS IN medicare_lakehouse

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN medicare_lakehouse.gold

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN medicare_lakehouse.ml_features

# COMMAND ----------

# MAGIC %md ## 3. Validate analyst views

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM medicare_lakehouse.gold.v_high_risk_summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM medicare_lakehouse.gold.v_cohort_trends LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM medicare_lakehouse.gold.v_top_hcc_conditions

# COMMAND ----------

# MAGIC %md ## 4. Table lineage via TBLPROPERTIES

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TBLPROPERTIES medicare_lakehouse.gold.member_raf_scores

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TBLPROPERTIES medicare_lakehouse.silver.hcc_mapped_claims

# COMMAND ----------

# MAGIC %md ## 5. Delta maintenance

# COMMAND ----------

from config.pipeline_config import Tables

maintenance_tables = [
    Tables.RAW_CLAIMS,
    Tables.CLEAN_CLAIMS,
    Tables.HCC_MAPPED,
    Tables.RAF_SCORES,
    Tables.FEATURE_STORE,
]

for table in maintenance_tables:
    try:
        spark.sql(f"OPTIMIZE {table}")
        print(f"✅ OPTIMIZE: {table}")
    except Exception as e:
        print(f"⚠️  OPTIMIZE skipped for {table}: {e}")

# COMMAND ----------

# Vacuum — retain 168 hours (7 days) per HIPAA-safe practice
for table in maintenance_tables:
    try:
        spark.sql(f"VACUUM {table} RETAIN 168 HOURS")
        print(f"✅ VACUUM: {table}")
    except Exception as e:
        print(f"⚠️  VACUUM skipped for {table}: {e}")

# COMMAND ----------

# MAGIC %md ## 6. Databricks Workflow Definition
# MAGIC
# MAGIC Copy the JSON below and import it via
# MAGIC **Workflows → Create Job → Edit JSON** in the Databricks UI.
# MAGIC
# MAGIC This defines a 4-task DAG:
# MAGIC ```
# MAGIC Bronze Ingestion
# MAGIC      ↓
# MAGIC Silver Processing
# MAGIC      ↓
# MAGIC Gold Aggregation
# MAGIC      ↓
# MAGIC Model Retrain & Drift Check
# MAGIC ```
# MAGIC Scheduled daily at 02:00 UTC.

# COMMAND ----------

import json

workflow_definition = {
    "name": "medicare_lakehouse_daily_pipeline",
    "description": "Medicare Claims Lakehouse: Bronze → Silver → Gold → Model Retrain",
    "schedule": {
        "quartz_cron_expression": "0 0 2 * * ?",
        "timezone_id": "UTC",
        "pause_status": "UNPAUSED"
    },
    "tasks": [
        {
            "task_key": "bronze_ingestion",
            "description": "Ingest synthetic CMS claims to Bronze Delta tables",
            "notebook_task": {
                "notebook_path": "/Repos/erickyegon/databricks-medicare-lakehouse/notebooks/01_bronze_ingestion",
                "base_parameters": {
                    "n_members":   "10000",
                    "n_months":    "24",
                    "batch_id":    "",
                    "output_path": "/tmp/medicare_lakehouse/raw"
                }
            },
            "existing_cluster_id": "{{cluster_id}}",
            "timeout_seconds": 1800,
            "max_retries": 2,
            "retry_on_timeout": True,
        },
        {
            "task_key": "silver_processing",
            "description": "Clean, deduplicate, and HCC-map Bronze claims",
            "depends_on": [{"task_key": "bronze_ingestion"}],
            "notebook_task": {
                "notebook_path": "/Repos/erickyegon/databricks-medicare-lakehouse/notebooks/02_silver_processing"
            },
            "existing_cluster_id": "{{cluster_id}}",
            "timeout_seconds": 2400,
            "max_retries": 1,
        },
        {
            "task_key": "gold_aggregation",
            "description": "Build RAF scores, utilization, trends, and feature store",
            "depends_on": [{"task_key": "silver_processing"}],
            "notebook_task": {
                "notebook_path": "/Repos/erickyegon/databricks-medicare-lakehouse/notebooks/03_gold_aggregates"
            },
            "existing_cluster_id": "{{cluster_id}}",
            "timeout_seconds": 2400,
            "max_retries": 1,
        },
        {
            "task_key": "model_retrain",
            "description": "Retrain XGBoost risk model and compute drift metrics",
            "depends_on": [{"task_key": "gold_aggregation"}],
            "notebook_task": {
                "notebook_path": "/Repos/erickyegon/databricks-medicare-lakehouse/notebooks/04_mlflow_model",
                "base_parameters": {
                    "run_name":        "",
                    "register_model": "yes"
                }
            },
            "existing_cluster_id": "{{cluster_id}}",
            "timeout_seconds": 3600,
            "max_retries": 1,
        },
    ],
    "email_notifications": {
        "on_failure": ["keyegon@gmail.com"],
        "on_success": [],
        "no_alert_for_skipped_runs": True,
    },
    "tags": {
        "project":     "medicare_lakehouse",
        "environment": "production",
        "owner":       "data_engineering",
        "hipaa":       "deidentified_synthetic",
    }
}

print(json.dumps(workflow_definition, indent=2))

# COMMAND ----------

# MAGIC %md ## 7. End-to-end lineage summary
# MAGIC
# MAGIC ```
# MAGIC CMS Synthetic Generator
# MAGIC        │
# MAGIC        ▼
# MAGIC bronze.raw_claims ──────────────────────── (append-only, partitioned year/month)
# MAGIC bronze.raw_members ─────────────────────── (full snapshot per batch)
# MAGIC        │
# MAGIC        ▼  [deduplicate + quality flags + date features]
# MAGIC silver.clean_claims ────────────────────── (quality_pass flag, 9 DQ flags)
# MAGIC        │
# MAGIC        ▼  [HCC v28 ICD-10 mapping + interaction terms]
# MAGIC silver.hcc_mapped_claims ───────────────── (hcc_list, raf_coeff, has_chf, etc.)
# MAGIC silver.member_profile ──────────────────── (demographics + demographic_raf)
# MAGIC        │
# MAGIC        ▼  [aggregate to member level]
# MAGIC gold.member_raf_scores ─────────────────── (raf_score, risk_tier, pmpm)
# MAGIC gold.utilization_summary ───────────────── (IP/ED/PMPM by member-period)
# MAGIC gold.monthly_trends ────────────────────── (DiD input: cohort × month × arm)
# MAGIC ml_features.risk_feature_store ─────────── (40 features, ML-ready)
# MAGIC        │
# MAGIC        ▼  [XGBoost + isotonic calibration + SHAP]
# MAGIC MLflow Model Registry
# MAGIC   └── medicare_raf_risk_model (Staging → Production)
# MAGIC        │
# MAGIC        ▼  [analyst-facing, no member IDs]
# MAGIC gold.v_high_risk_summary
# MAGIC gold.v_cohort_trends
# MAGIC gold.v_top_hcc_conditions
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## ✅ Pipeline fully operational
# MAGIC
# MAGIC | Component | Status |
# MAGIC |-----------|--------|
# MAGIC | Bronze Delta tables | ✅ Partitioned, optimized |
# MAGIC | Silver cleaning + HCC mapping | ✅ Quality-validated |
# MAGIC | Gold aggregates + feature store | ✅ RAF scores, DiD input, ML features |
# MAGIC | MLflow experiment tracking | ✅ Metrics, SHAP, model registry |
# MAGIC | Unity Catalog governance | ✅ GRANT/REVOKE, lineage tags, analyst views |
# MAGIC | Databricks Workflow | ✅ 4-task DAG, daily 02:00 UTC schedule |
