"""
src/silver/clean_claims.py
──────────────────────────
Silver layer: clean, deduplicate, enrich, and HCC-map Bronze claims.
Produces two Silver tables:
  - clean_claims:      deduplicated, validated claim lines
  - hcc_mapped_claims: clean_claims enriched with HCC/RAF columns
  - member_profile:    demographics joined from Bronze members
"""

from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

from config.pipeline_config import CONFIG, SilverConfig
from src.silver.hcc_mapper import HCCMapper, get_demographic_raf
from src.utils.spark_utils import (
    add_audit_columns, flag_quality_issues, get_logger,
    upsert_delta, write_delta, get_spark,
)

logger = get_logger(__name__)


class SilverProcessor:
    """
    Silver layer transformation pipeline.

    Stages (in order):
      1. Cast + parse raw types
      2. Deduplication
      3. Quality flagging and filtering
      4. Feature extraction (date parts, service categories)
      5. HCC / RAF mapping
      6. Member profile join
      7. Write to Silver Delta tables

    Usage:
        processor = SilverProcessor(spark)
        processor.run()
    """

    def __init__(
        self,
        spark:  Optional[SparkSession] = None,
        config: Optional[SilverConfig] = None,
    ):
        self.spark  = spark or get_spark()
        self.config = config or CONFIG.silver
        self.mapper = HCCMapper(self.spark)

    # ── Stage 1: Type casting ───────────────────────────────────────────

    def _cast_types(self, df: DataFrame) -> DataFrame:
        """Cast string columns from Bronze to proper Spark types."""
        return (
            df
            .withColumn("service_date",  F.to_date("service_date", "yyyy-MM-dd"))
            .withColumn("claim_amount",  F.col("claim_amount").cast("double"))
            .withColumn("allowed_amount",F.col("allowed_amount").cast("double"))
            .withColumn("paid_amount",   F.col("paid_amount").cast("double"))
            .withColumn("claim_year",    F.year("service_date"))
            .withColumn("claim_month",   F.month("service_date"))
        )

    # ── Stage 2: Deduplication ──────────────────────────────────────────

    def _deduplicate(self, df: DataFrame) -> DataFrame:
        """
        Remove duplicate claims using the configured dedup keys.
        Keeps the row with the highest paid_amount when duplicates exist.
        """
        n_before = df.count()
        df = (
            df
            .withColumn(
                "_row_rank",
                F.row_number().over(
                    __import__("pyspark.sql.window", fromlist=["Window"])
                    .Window
                    .partitionBy(*self.config.dedup_keys)
                    .orderBy(F.desc("paid_amount"))
                )
            )
            .filter(F.col("_row_rank") == 1)
            .drop("_row_rank")
        )
        n_after = df.count()
        logger.info(f"Deduplication: {n_before:,} → {n_after:,} rows ({n_before - n_after:,} dropped)")
        return df

    # ── Stage 3: Quality flagging ───────────────────────────────────────

    def _flag_quality(self, df: DataFrame) -> DataFrame:
        """
        Add quality flags. Rows with _quality_pass=False are excluded from
        HCC mapping and Gold but are retained in clean_claims for auditing.
        """
        rules = {
            "flag_null_claim_id":     F.col("claim_id").isNull(),
            "flag_null_bene_id":      F.col("bene_id").isNull(),
            "flag_negative_amount":   F.col("claim_amount") < 0,
            "flag_zero_amount":       F.col("claim_amount") == 0,
            "flag_excessive_amount":  F.col("claim_amount") > self.config.max_claim_amount,
            "flag_future_date":       F.col("service_date") > F.current_date(),
            "flag_null_service_date": F.col("service_date").isNull(),
            "flag_null_icd10":        F.col("icd10_primary").isNull(),
            "flag_bronze_null":       F.col("_bronze_null_flag") == True,
        }
        df = flag_quality_issues(df, rules)

        n_fail = df.filter(~F.col("_quality_pass")).count()
        n_total = df.count()
        logger.info(
            f"Quality check: {n_fail:,} / {n_total:,} rows flagged "
            f"({n_fail / max(n_total, 1) * 100:.1f}% fail rate)"
        )
        return df

    # ── Stage 4: Feature extraction ─────────────────────────────────────

    def _extract_features(self, df: DataFrame) -> DataFrame:
        """
        Extract date-based and service-category features from claims.
        These feed directly into the Gold feature store.
        """
        return (
            df
            .withColumn("service_year",      F.year("service_date"))
            .withColumn("service_month",     F.month("service_date"))
            .withColumn("service_quarter",   F.quarter("service_date"))
            .withColumn("day_of_week",       F.dayofweek("service_date"))
            .withColumn("is_weekend",        F.dayofweek("service_date").isin([1, 7]))
            .withColumn("period",            F.when(
                F.col("service_year") >= 2023, "post"
            ).otherwise("pre"))
            # Service type categories
            .withColumn("is_inpatient",      F.col("service_type") == "ip_admit")
            .withColumn("is_ed_visit",       F.col("service_type") == "ed_visit")
            .withColumn("is_specialist",     F.col("service_type") == "specialist")
            .withColumn("is_primary_care",   F.col("service_type") == "primary")
            # Cost tiers
            .withColumn("cost_tier", F.when(
                F.col("claim_amount") < 500,   "low"
            ).when(
                F.col("claim_amount") < 5_000, "medium"
            ).otherwise("high"))
        )

    # ── Stage 5: Member profile ─────────────────────────────────────────

    def _build_member_profile(self) -> DataFrame:
        """
        Build a clean member profile table from Bronze members.
        Adds demographic RAF coefficient.
        """
        members = self.spark.table(self.config.bronze_members_table)

        # Cast types
        members = (
            members
            .withColumn("enrollment_date", F.to_date("enrollment_date", "yyyy-MM-dd"))
            .withColumn("age",             F.col("age").cast("int"))
            .withColumn("dual_eligible",   F.col("dual_eligible").cast("int"))
        )

        # Demographic RAF UDF
        get_demo_raf = F.udf(get_demographic_raf, __import__("pyspark.sql.types", fromlist=["DoubleType"]).DoubleType())

        members = (
            members
            .withColumn("demographic_raf", get_demo_raf(F.col("sex"), F.col("age")))
            .withColumn("age_bracket", (F.floor(F.col("age") / 5) * 5).cast("int"))
            .withColumn("is_aged",    F.col("age") >= 65)
        )

        # Quality flags for members
        members = (
            members
            .withColumn("flag_invalid_age", (F.col("age") < 0) | (F.col("age") > 120))
            .withColumn("flag_null_sex",    F.col("sex").isNull())
        )

        logger.info(f"Member profile built: {members.count():,} members")
        return members

    # ── Main runner ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full Silver processing pipeline.

        Returns:
            dict with row counts and table names.
        """
        logger.info("Silver processing starting")

        # Ensure Silver schema exists
        from config.pipeline_config import CATALOG, SILVER_SCHEMA
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}")

        # ── Load Bronze claims ────────────────────────────────────────
        logger.info(f"Loading Bronze claims: {self.config.bronze_claims_table}")
        raw_claims = self.spark.table(self.config.bronze_claims_table)

        # ── Transform ─────────────────────────────────────────────────
        clean = raw_claims
        clean = self._cast_types(clean)
        clean = self._deduplicate(clean)
        clean = self._flag_quality(clean)
        clean = self._extract_features(clean)
        clean = add_audit_columns(clean, "silver")

        # ── Write clean claims ────────────────────────────────────────
        clean_count = write_delta(
            clean,
            table_name=self.config.clean_claims_table,
            mode="overwrite",
            partition_by=["service_year", "service_month"],
            optimize_after=True,
            zorder_cols=["bene_id"],
            spark=self.spark,
        )

        # ── HCC mapping (only quality-pass rows) ──────────────────────
        logger.info("Applying HCC mapping to quality-pass claims")
        quality_claims = clean.filter(F.col("_quality_pass") == True)
        hcc_mapped     = self.mapper.map(quality_claims)
        hcc_mapped     = add_audit_columns(hcc_mapped, "silver")

        hcc_count = write_delta(
            hcc_mapped,
            table_name=self.config.hcc_mapped_table,
            mode="overwrite",
            partition_by=["service_year", "service_month"],
            optimize_after=True,
            zorder_cols=["bene_id", "hcc_burden_count"],
            spark=self.spark,
        )

        # ── Member profile ─────────────────────────────────────────────
        member_profile  = self._build_member_profile()
        member_profile  = add_audit_columns(member_profile, "silver")
        member_count    = write_delta(
            member_profile,
            table_name=self.config.member_profile_table,
            mode="overwrite",
            spark=self.spark,
        )

        result = {
            "clean_claims_rows":  clean_count,
            "hcc_mapped_rows":    hcc_count,
            "member_profile_rows":member_count,
            "tables": {
                "clean_claims":    self.config.clean_claims_table,
                "hcc_mapped":      self.config.hcc_mapped_table,
                "member_profile":  self.config.member_profile_table,
            }
        }
        logger.info(
            f"Silver complete — clean: {clean_count:,} | "
            f"hcc_mapped: {hcc_count:,} | members: {member_count:,}"
        )
        return result


def run_silver_processing(
    spark: Optional[SparkSession] = None,
) -> dict:
    """One-line entry point for Databricks notebooks."""
    return SilverProcessor(spark=spark).run()
