# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 01 — Bronze Ingestion
# MAGIC
# MAGIC **Layer**: Bronze (raw ingestion)
# MAGIC **Purpose**: Generate synthetic CMS claims data and ingest into append-only Delta tables
# MAGIC
# MAGIC ## What this notebook does
# MAGIC 1. Accepts widgets for `n_members`, `n_months`, and `batch_id`
# MAGIC 2. Generates synthetic CMS-style claims and member demographics
# MAGIC 3. Saves raw files to the landing zone volume
# MAGIC 4. Ingests to Bronze Delta tables with schema enforcement and audit columns
# MAGIC 5. Runs Delta OPTIMIZE + ZORDER for downstream query performance
# MAGIC
# MAGIC **Expected runtime**: ~3–5 minutes for N=10,000 members, 24 months

# COMMAND ----------

# MAGIC %md ## 0. Parameters

# COMMAND ----------

dbutils.widgets.text("n_members",   "10000",  "Number of synthetic members")
dbutils.widgets.text("n_months",    "24",     "Months of claims history")
dbutils.widgets.text("batch_id",    "",       "Batch ID (leave blank for auto)")
dbutils.widgets.text("output_path", "/tmp/medicare_lakehouse/raw", "Landing zone path")

N_MEMBERS   = int(dbutils.widgets.get("n_members"))
N_MONTHS    = int(dbutils.widgets.get("n_months"))
BATCH_ID    = dbutils.widgets.get("batch_id") or None
OUTPUT_PATH = dbutils.widgets.get("output_path")

print(f"Config: N_MEMBERS={N_MEMBERS:,} | N_MONTHS={N_MONTHS} | BATCH_ID={BATCH_ID or 'auto'}")

# COMMAND ----------

# MAGIC %md ## 1. Setup: add src/ to Python path

# COMMAND ----------

import sys, os
repo_root = os.path.abspath("..")   # adjust if running from /notebooks/
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md ## 2. Generate synthetic CMS data

# COMMAND ----------

from src.data_generator.cms_claims_generator import generate_and_save

members_df, claims_df = generate_and_save(
    output_dir=OUTPUT_PATH,
    n_members=N_MEMBERS,
    n_months=N_MONTHS,
    intervention_effect=-420.0,
    seed=42,
)

print(f"✅ Generated {len(members_df):,} members | {len(claims_df):,} claim lines")
display(members_df.head(5))

# COMMAND ----------

# MAGIC %md ### Claims sample

# COMMAND ----------

display(claims_df.head(10))

# COMMAND ----------

# MAGIC %md ### Data quality snapshot

# COMMAND ----------

print("=== Members ===")
print(members_df.dtypes)
print(f"\nIntervention arm distribution:\n{members_df['intervention_arm'].value_counts()}")

print("\n=== Claims ===")
print(f"Service types:\n{claims_df['service_type'].value_counts()}")
print(f"\nClaim amount stats:\n{claims_df['claim_amount'].describe().round(2)}")

# COMMAND ----------

# MAGIC %md ## 3. Run Bronze Ingestion

# COMMAND ----------

from src.bronze.ingest_claims import run_bronze_ingestion

result = run_bronze_ingestion(
    claims_path  = f"{OUTPUT_PATH}/claims.csv",
    members_path = f"{OUTPUT_PATH}/members.csv",
    batch_id     = BATCH_ID,
    spark        = spark,
)

print("✅ Bronze ingestion complete")
print(f"   Claims table:  {result['claims_table']}")
print(f"   Members table: {result['members_table']}")
print(f"   Claims rows:   {result['claims_rows']:,}")
print(f"   Members rows:  {result['members_rows']:,}")
print(f"   Batch ID:      {result['batch_id']}")

# COMMAND ----------

# MAGIC %md ## 4. Verify Bronze tables

# COMMAND ----------

from config.pipeline_config import Tables

print("=== Bronze Claims ===")
bronze_claims = spark.table(Tables.RAW_CLAIMS)
print(f"Row count:  {bronze_claims.count():,}")
print(f"Partitions: {bronze_claims.rdd.getNumPartitions()}")
bronze_claims.printSchema()

# COMMAND ----------

display(bronze_claims.limit(5))

# COMMAND ----------

# MAGIC %md ### Time travel check — Delta version history

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY medicare_lakehouse.bronze.raw_claims LIMIT 5

# COMMAND ----------

# MAGIC %md ### Partition stats

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   claim_year,
# MAGIC   claim_month,
# MAGIC   COUNT(*) AS n_claims,
# MAGIC   ROUND(SUM(claim_amount), 0) AS total_amount
# MAGIC FROM medicare_lakehouse.bronze.raw_claims
# MAGIC GROUP BY claim_year, claim_month
# MAGIC ORDER BY claim_year, claim_month

# COMMAND ----------

# MAGIC %md ## 5. Bronze quality summary

# COMMAND ----------

from pyspark.sql import functions as F

null_summary = bronze_claims.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in ["claim_id", "bene_id", "service_date", "claim_amount", "icd10_primary"]
])
display(null_summary)

# COMMAND ----------

null_flag_count = bronze_claims.filter(F.col("_bronze_null_flag") == True).count()
print(f"Rows with critical nulls (_bronze_null_flag=True): {null_flag_count:,}")
print(f"These will be excluded in Silver processing.")

# COMMAND ----------

# MAGIC %md ## ✅ Bronze complete
# MAGIC
# MAGIC | Table | Rows | Status |
# MAGIC |-------|------|--------|
# MAGIC | `bronze.raw_claims` | See above | ✅ Partitioned by year/month |
# MAGIC | `bronze.raw_members` | See above | ✅ Full snapshot |
# MAGIC
# MAGIC **Next**: Run `02_silver_processing` to clean, deduplicate, and HCC-map the claims.
