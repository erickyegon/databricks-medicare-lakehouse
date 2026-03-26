# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02 — Silver Processing
# MAGIC
# MAGIC **Layer**: Silver (clean, enrich, HCC-map)
# MAGIC **Inputs**: `bronze.raw_claims`, `bronze.raw_members`
# MAGIC **Outputs**:
# MAGIC   - `silver.clean_claims` — deduplicated, quality-flagged claims
# MAGIC   - `silver.hcc_mapped_claims` — HCC v28 enriched claims with RAF coefficients
# MAGIC   - `silver.member_profile` — cleaned demographics with demographic RAF
# MAGIC
# MAGIC **Expected runtime**: ~5–8 minutes for 10K members / 24 months

# COMMAND ----------

import sys, os
repo_root = os.path.abspath("..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md ## 1. Run Silver processor

# COMMAND ----------

from src.silver.clean_claims import run_silver_processing

result = run_silver_processing(spark=spark)

print("✅ Silver processing complete")
for k, v in result["tables"].items():
    print(f"   {k}: {result.get(k+'_rows', '?'):,} rows → {v}")

# COMMAND ----------

# MAGIC %md ## 2. Quality validation

# COMMAND ----------

from config.pipeline_config import Tables
from pyspark.sql import functions as F

clean = spark.table(Tables.CLEAN_CLAIMS)
total = clean.count()
pass_count = clean.filter(F.col("_quality_pass") == True).count()
fail_count = total - pass_count

print(f"Total rows:      {total:,}")
print(f"Quality pass:    {pass_count:,} ({pass_count/total*100:.1f}%)")
print(f"Quality fail:    {fail_count:,} ({fail_count/total*100:.1f}%)")

# COMMAND ----------

# MAGIC %md ### Quality flag breakdown

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   SUM(CAST(flag_null_claim_id    AS INT)) AS null_claim_id,
# MAGIC   SUM(CAST(flag_null_bene_id     AS INT)) AS null_bene_id,
# MAGIC   SUM(CAST(flag_negative_amount  AS INT)) AS negative_amount,
# MAGIC   SUM(CAST(flag_zero_amount      AS INT)) AS zero_amount,
# MAGIC   SUM(CAST(flag_excessive_amount AS INT)) AS excessive_amount,
# MAGIC   SUM(CAST(flag_future_date      AS INT)) AS future_date,
# MAGIC   SUM(CAST(flag_null_service_date AS INT)) AS null_service_date,
# MAGIC   SUM(CAST(flag_null_icd10       AS INT)) AS null_icd10,
# MAGIC   COUNT(*) AS total_rows
# MAGIC FROM medicare_lakehouse.silver.clean_claims

# COMMAND ----------

# MAGIC %md ## 3. HCC mapping validation

# COMMAND ----------

hcc = spark.table(Tables.HCC_MAPPED)
print(f"HCC-mapped rows: {hcc.count():,}")

# HCC burden distribution
display(
    hcc.groupBy("hcc_burden_count")
    .count()
    .orderBy("hcc_burden_count")
)

# COMMAND ----------

# MAGIC %md ### Top HCC conditions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   primary_hcc_desc,
# MAGIC   primary_hcc,
# MAGIC   COUNT(DISTINCT bene_id)   AS n_members,
# MAGIC   COUNT(*)                  AS n_claims,
# MAGIC   ROUND(AVG(hcc_raf_total), 3) AS mean_hcc_raf
# MAGIC FROM medicare_lakehouse.silver.hcc_mapped_claims
# MAGIC WHERE primary_hcc_desc IS NOT NULL
# MAGIC GROUP BY primary_hcc_desc, primary_hcc
# MAGIC ORDER BY n_members DESC
# MAGIC LIMIT 15

# COMMAND ----------

# MAGIC %md ### Condition co-occurrence flags

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   SUM(CAST(has_chf      AS INT)) AS n_chf,
# MAGIC   SUM(CAST(has_afib     AS INT)) AS n_afib,
# MAGIC   SUM(CAST(has_diabetes AS INT)) AS n_diabetes,
# MAGIC   SUM(CAST(has_ckd      AS INT)) AS n_ckd,
# MAGIC   SUM(CAST(has_cancer   AS INT)) AS n_cancer,
# MAGIC   SUM(CAST(has_copd     AS INT)) AS n_copd,
# MAGIC   SUM(CAST(has_metastatic AS INT)) AS n_metastatic,
# MAGIC   COUNT(DISTINCT bene_id) AS total_members
# MAGIC FROM medicare_lakehouse.silver.hcc_mapped_claims

# COMMAND ----------

# MAGIC %md ## 4. Member profile

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   sex,
# MAGIC   dual_eligible,
# MAGIC   COUNT(*)                  AS n_members,
# MAGIC   ROUND(AVG(age), 1)        AS mean_age,
# MAGIC   ROUND(AVG(demographic_raf), 3) AS mean_demo_raf
# MAGIC FROM medicare_lakehouse.silver.member_profile
# MAGIC GROUP BY sex, dual_eligible
# MAGIC ORDER BY sex, dual_eligible

# COMMAND ----------

# MAGIC %md ## 5. Delta time travel — query last version

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY medicare_lakehouse.silver.hcc_mapped_claims LIMIT 3

# COMMAND ----------

# Read previous version (time travel)
prev_version = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .table(Tables.HCC_MAPPED)
print(f"Version 0 row count: {prev_version.count():,}")

# COMMAND ----------

# MAGIC %md ## ✅ Silver complete
# MAGIC
# MAGIC | Table | Rows | Notes |
# MAGIC |-------|------|-------|
# MAGIC | `silver.clean_claims` | See above | Deduplicated, quality-flagged |
# MAGIC | `silver.hcc_mapped_claims` | See above | HCC v28 enriched, RAF coefficients |
# MAGIC | `silver.member_profile` | See above | Demographics + demographic RAF |
# MAGIC
# MAGIC **Next**: Run `03_gold_aggregates` to build member-level RAF scores and the ML feature store.
