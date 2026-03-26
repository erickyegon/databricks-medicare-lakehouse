# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03 — Gold Aggregates & Feature Store
# MAGIC
# MAGIC **Layer**: Gold (analytics-ready aggregates)
# MAGIC **Inputs**: `silver.hcc_mapped_claims`, `silver.member_profile`, `silver.clean_claims`
# MAGIC **Outputs**:
# MAGIC   - `gold.member_raf_scores`  — final RAF score, risk tier, cost estimates
# MAGIC   - `gold.utilization_summary` — IP/ED/PMPM by member-period
# MAGIC   - `gold.monthly_trends`     — cohort-level monthly aggregates for DiD analysis
# MAGIC   - `ml_features.risk_feature_store` — ML-ready feature matrix
# MAGIC
# MAGIC **Expected runtime**: ~5–10 minutes

# COMMAND ----------

import sys, os
repo_root = os.path.abspath("..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md ## 1. Run Gold aggregation

# COMMAND ----------

from src.gold.raf_aggregates import run_gold_aggregation

result = run_gold_aggregation(spark=spark)

print("✅ Gold aggregation complete")
print(f"   Mean RAF score: {result['mean_raf']}")
print(f"   Risk tier distribution: {result['tier_counts']}")
for k, v in result["tables"].items():
    print(f"   {k}: {v}")

# COMMAND ----------

# MAGIC %md ## 2. RAF score distribution

# COMMAND ----------

from config.pipeline_config import Tables
from pyspark.sql import functions as F

raf = spark.table(Tables.RAF_SCORES)
display(raf.select("bene_id", "raf_score", "risk_tier", "estimated_annual_cost",
                   "actual_pmpm", "ip_admit_count", "ed_visit_count").limit(20))

# COMMAND ----------

# MAGIC %md ### RAF distribution by tier

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   risk_tier,
# MAGIC   COUNT(*)                          AS n_members,
# MAGIC   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct_members,
# MAGIC   ROUND(AVG(raf_score), 3)          AS mean_raf,
# MAGIC   ROUND(MIN(raf_score), 3)          AS min_raf,
# MAGIC   ROUND(MAX(raf_score), 3)          AS max_raf,
# MAGIC   ROUND(AVG(estimated_annual_cost), 0) AS mean_est_cost,
# MAGIC   ROUND(AVG(actual_pmpm), 2)        AS mean_actual_pmpm,
# MAGIC   SUM(ip_admit_count)               AS total_ip_admits,
# MAGIC   SUM(ed_visit_count)               AS total_ed_visits
# MAGIC FROM medicare_lakehouse.gold.member_raf_scores
# MAGIC GROUP BY risk_tier
# MAGIC ORDER BY mean_raf DESC

# COMMAND ----------

# MAGIC %md ## 3. Utilization summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   period,
# MAGIC   intervention_arm,
# MAGIC   COUNT(DISTINCT bene_id)       AS n_members,
# MAGIC   ROUND(AVG(pmpm_cost), 2)      AS mean_pmpm,
# MAGIC   ROUND(AVG(ip_rate_per_1000), 1) AS ip_rate_1000,
# MAGIC   ROUND(AVG(ed_rate_per_1000), 1) AS ed_rate_1000,
# MAGIC   SUM(ip_admits)                AS total_ip,
# MAGIC   SUM(ed_visits)                AS total_ed
# MAGIC FROM medicare_lakehouse.gold.utilization_summary
# MAGIC GROUP BY period, intervention_arm
# MAGIC ORDER BY period DESC, intervention_arm

# COMMAND ----------

# MAGIC %md ## 4. DiD-style monthly trends

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   service_year,
# MAGIC   service_month,
# MAGIC   intervention_arm,
# MAGIC   n_members,
# MAGIC   ROUND(pmpm_cost, 2)       AS pmpm_cost,
# MAGIC   ROUND(pmpm_rolling_3m, 2) AS pmpm_3m_avg,
# MAGIC   period
# MAGIC FROM medicare_lakehouse.gold.monthly_trends
# MAGIC ORDER BY service_year, service_month, intervention_arm

# COMMAND ----------

# MAGIC %md ### DiD estimate (manual check)

# COMMAND ----------

from pyspark.sql import functions as F

trends = spark.table(Tables.MONTHLY_TRENDS)

# Pre and post means by intervention arm
did_check = (
    trends
    .groupBy("period", "intervention_arm")
    .agg(F.mean("pmpm_cost").alias("mean_pmpm"))
    .orderBy("period", "intervention_arm")
)
display(did_check)

# Manual DiD calculation
rows = {(r["period"], r["intervention_arm"]): r["mean_pmpm"]
        for r in did_check.collect()}

pre_control  = rows.get(("pre",  0), 0)
pre_treated  = rows.get(("pre",  1), 0)
post_control = rows.get(("post", 0), 0)
post_treated = rows.get(("post", 1), 0)

did_estimate = (post_treated - pre_treated) - (post_control - pre_control)
print(f"\nDiD estimate (ATT): ${did_estimate:,.2f}/member/month")
print(f"Annualized ATT:      ${did_estimate * 12:,.0f}/member/year")

# COMMAND ----------

# MAGIC %md ## 5. ML Feature Store preview

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   risk_tier,
# MAGIC   COUNT(*)                  AS n_members,
# MAGIC   ROUND(AVG(raf_score), 3)  AS mean_raf,
# MAGIC   ROUND(AVG(pre_pmpm), 2)   AS mean_pre_pmpm,
# MAGIC   ROUND(AVG(pre_ip_admits), 2) AS mean_pre_ip,
# MAGIC   ROUND(AVG(max_hcc_burden), 1) AS mean_hcc_burden
# MAGIC FROM medicare_lakehouse.ml_features.risk_feature_store
# MAGIC GROUP BY risk_tier
# MAGIC ORDER BY mean_raf DESC

# COMMAND ----------

# MAGIC %md ### Feature correlation heatmap (Pandas)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

feat_pd = spark.table(Tables.FEATURE_STORE).toPandas()

numeric_feats = [
    "age", "demographic_raf", "max_hcc_burden", "max_hcc_raf",
    "pre_pmpm", "pre_ip_admits", "pre_ed_visits", "raf_score"
]
corr = feat_pd[numeric_feats].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix — Gold Feature Store")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md ## 6. MSSP Shared Savings Projection

# COMMAND ----------

from src.gold.shared_savings import SharedSavingsCalculator

calc = SharedSavingsCalculator()

# Compute from Gold actuals
result = calc.compute_from_gold(spark)
print(result)

# COMMAND ----------

# MAGIC %md ### Scale projection — ATT from companion RAF pipeline (−$391/member/year)

# COMMAND ----------

projection = calc.project(att_pmpm=391.0)
display(projection.reset_index())

# COMMAND ----------

# MAGIC %md ### Interactive scenario — adjust sharing rate and MSR

# COMMAND ----------

dbutils.widgets.text("att_pmpm",     "391",  "ATT $/member/year")
dbutils.widgets.text("sharing_rate", "0.50", "MSSP sharing rate")
dbutils.widgets.text("n_lives",      "10000","Attributed lives")

att      = float(dbutils.widgets.get("att_pmpm"))
sr       = float(dbutils.widgets.get("sharing_rate"))
n        = int(dbutils.widgets.get("n_lives"))

scenario = calc.from_att(att_pmpm_annual=att, n_lives=n, sharing_rate=sr)
print(scenario)

# COMMAND ----------

# MAGIC %md ## ✅ Gold complete
# MAGIC
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `gold.member_raf_scores` | Final RAF score, risk tier, cost per member |
# MAGIC | `gold.utilization_summary` | IP/ED/PMPM by member-period |
# MAGIC | `gold.monthly_trends` | Cohort-level DiD input |
# MAGIC | `ml_features.risk_feature_store` | XGBoost training input |
# MAGIC
# MAGIC **Next**: Run `04_mlflow_model` to train and register the XGBoost risk model.
