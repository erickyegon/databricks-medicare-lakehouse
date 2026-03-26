"""
src/bronze/ingest_claims.py
───────────────────────────
Bronze layer: raw ingestion of CMS claims and member files into
append-only Delta tables with schema enforcement and audit columns.

Design decisions:
  - Append-only: never update or delete Bronze rows — full audit trail
  - Schema enforcement: reject malformed files before they corrupt downstream
  - Partition by claim_year / claim_month for efficient Silver reads
  - Idempotent: re-running the same batch does not duplicate data
    (uses _batch_id + MERGE key to prevent re-ingestion)
"""

from datetime import datetime
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, LongType, StringType, StructField, StructType
)

from config.pipeline_config import CONFIG, BronzeConfig
from src.utils.spark_utils import (
    add_audit_columns, get_logger, table_exists,
    validate_schema, write_delta, get_spark,
)

logger = get_logger(__name__)


# ── Expected schemas ───────────────────────────────────────────────────────

CLAIMS_SCHEMA = StructType([
    StructField("claim_id",           StringType(),  False),
    StructField("bene_id",            StringType(),  False),
    StructField("service_date",       StringType(),  False),  # cast to date in Silver
    StructField("claim_year",         IntegerType(), True),
    StructField("claim_month",        IntegerType(), True),
    StructField("service_type",       StringType(),  True),
    StructField("icd10_primary",      StringType(),  True),
    StructField("icd10_codes",        StringType(),  True),   # pipe-delimited
    StructField("provider_specialty", StringType(),  True),
    StructField("claim_amount",       DoubleType(),  True),
    StructField("allowed_amount",     DoubleType(),  True),
    StructField("paid_amount",        DoubleType(),  True),
    StructField("plan_type",          StringType(),  True),
    StructField("intervention_arm",   IntegerType(), True),
])

MEMBERS_SCHEMA = StructType([
    StructField("bene_id",           StringType(),  False),
    StructField("age",               IntegerType(), True),
    StructField("sex",               StringType(),  True),
    StructField("dual_eligible",     IntegerType(), True),
    StructField("plan_type",         StringType(),  True),
    StructField("state",             StringType(),  True),
    StructField("enrollment_date",   StringType(),  True),
    StructField("intervention_arm",  IntegerType(), True),
])


# ── Loader ─────────────────────────────────────────────────────────────────

class BronzeIngestion:
    """
    Handles reading raw CSV/Parquet files and writing to Bronze Delta tables.

    Usage:
        ingestion = BronzeIngestion(spark, config=CONFIG.bronze)
        ingestion.run(claims_path="...", members_path="...")
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        config: Optional[BronzeConfig] = None,
    ):
        self.spark  = spark or get_spark()
        self.config = config or CONFIG.bronze

    # ── Readers ────────────────────────────────────────────────────────

    def _read_csv(self, path: str, schema: StructType) -> DataFrame:
        """Read a CSV file into a Spark DataFrame with strict schema."""
        logger.info(f"Reading CSV: {path}")
        df = (
            self.spark.read
            .option("header", "true")
            .option("mode", "PERMISSIVE")       # log malformed rows
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .schema(schema)
            .csv(path)
        )
        return df

    def _read_parquet(self, path: str) -> DataFrame:
        """Read a Parquet file (for pre-generated synthetic data)."""
        logger.info(f"Reading Parquet: {path}")
        return self.spark.read.parquet(path)

    def _read_auto(self, path: str, schema: StructType) -> DataFrame:
        """Auto-detect file format from extension."""
        if path.endswith(".parquet"):
            return self._read_parquet(path)
        return self._read_csv(path, schema)

    # ── Validation ─────────────────────────────────────────────────────

    def _validate_claims(self, df: DataFrame) -> DataFrame:
        """
        Run Bronze-level validation on claims.
        At this layer we only log issues — we do NOT drop rows.
        All raw data must land in Bronze, even dirty rows.
        """
        issues = validate_schema(df, CLAIMS_SCHEMA, strict=False)
        if issues:
            for issue in issues:
                logger.warning(f"Schema issue: {issue}")

        # Count nulls on key columns
        n_null_claim_id = df.filter(F.col("claim_id").isNull()).count()
        n_null_bene_id  = df.filter(F.col("bene_id").isNull()).count()
        n_null_amount   = df.filter(F.col("claim_amount").isNull()).count()

        logger.info(
            f"Bronze validation — "
            f"null claim_id: {n_null_claim_id:,} | "
            f"null bene_id: {n_null_bene_id:,} | "
            f"null claim_amount: {n_null_amount:,}"
        )

        # Tag rows that have critical nulls (Silver will filter these)
        df = df.withColumn(
            "_bronze_null_flag",
            F.col("claim_id").isNull() | F.col("bene_id").isNull()
        )
        return df

    def _validate_members(self, df: DataFrame) -> DataFrame:
        """Bronze-level member validation — log only, no drops."""
        n_null_bene = df.filter(F.col("bene_id").isNull()).count()
        logger.info(f"Bronze validation — null bene_id in members: {n_null_bene:,}")
        df = df.withColumn("_bronze_null_flag", F.col("bene_id").isNull())
        return df

    # ── Idempotency guard ───────────────────────────────────────────────

    def _batch_already_ingested(self, table_name: str, batch_id: str) -> bool:
        """Return True if this batch_id already exists in the target table."""
        if not table_exists(self.spark, table_name):
            return False
        count = (
            self.spark.table(table_name)
            .filter(F.col("_batch_id") == batch_id)
            .limit(1)
            .count()
        )
        return count > 0

    # ── Writers ─────────────────────────────────────────────────────────

    def _write_bronze_claims(self, df: DataFrame, batch_id: str) -> int:
        """Write claims to Bronze Delta table, partitioned by year/month."""
        if self._batch_already_ingested(self.config.claims_table, batch_id):
            logger.warning(f"Batch {batch_id} already in {self.config.claims_table} — skipping")
            return 0

        df = add_audit_columns(df, "bronze")
        df = df.withColumn("_batch_id", F.lit(batch_id))  # override auto batch_id

        return write_delta(
            df,
            table_name=self.config.claims_table,
            mode="append",
            partition_by=["claim_year", "claim_month"],
            optimize_after=True,
            zorder_cols=["bene_id"],
            spark=self.spark,
        )

    def _write_bronze_members(self, df: DataFrame, batch_id: str) -> int:
        """Write members to Bronze Delta table (overwrite — members are a snapshot)."""
        df = add_audit_columns(df, "bronze")
        df = df.withColumn("_batch_id", F.lit(batch_id))

        return write_delta(
            df,
            table_name=self.config.members_table,
            mode="overwrite",
            spark=self.spark,
        )

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_database(self) -> None:
        """Ensure the Bronze schema exists in Unity Catalog."""
        from config.pipeline_config import CATALOG, BRONZE_SCHEMA
        self.spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
        self.spark.sql(f"USE CATALOG {CATALOG}")
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {BRONZE_SCHEMA}")
        logger.info(f"Bronze schema ready: {CATALOG}.{BRONZE_SCHEMA}")

    # ── Public entry point ───────────────────────────────────────────────

    def run(
        self,
        claims_path:  str,
        members_path: str,
        batch_id:     Optional[str] = None,
    ) -> dict:
        """
        Ingest claims and member files into Bronze Delta tables.

        Args:
            claims_path:  Path to claims CSV/Parquet.
            members_path: Path to members CSV/Parquet.
            batch_id:     Override batch ID (default: yyyyMMddHHmmss).

        Returns:
            dict with row counts and table names.
        """
        batch_id = batch_id or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        logger.info(f"Bronze ingestion starting — batch_id={batch_id}")

        self._setup_database()

        # Claims
        claims_df   = self._read_auto(claims_path, CLAIMS_SCHEMA)
        claims_df   = self._validate_claims(claims_df)
        claims_rows = self._write_bronze_claims(claims_df, batch_id)

        # Members
        members_df   = self._read_auto(members_path, MEMBERS_SCHEMA)
        members_df   = self._validate_members(members_df)
        members_rows = self._write_bronze_members(members_df, batch_id)

        result = {
            "batch_id":     batch_id,
            "claims_table":  self.config.claims_table,
            "members_table": self.config.members_table,
            "claims_rows":   claims_rows,
            "members_rows":  members_rows,
        }

        logger.info(
            f"Bronze ingestion complete — "
            f"claims: {claims_rows:,} rows | members: {members_rows:,} rows"
        )
        return result


# ── Convenience function for notebook use ──────────────────────────────────

def run_bronze_ingestion(
    claims_path:  str,
    members_path: str,
    batch_id:     Optional[str] = None,
    spark:        Optional[SparkSession] = None,
) -> dict:
    """One-line entry point for Databricks notebooks."""
    return BronzeIngestion(spark=spark).run(
        claims_path=claims_path,
        members_path=members_path,
        batch_id=batch_id,
    )
