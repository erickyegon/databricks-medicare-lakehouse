"""
src/gold/raf_aggregates.py
──────────────────────────
Gold layer: member-level RAF scores, utilization summaries, monthly trends,
and an ML-ready feature store for the XGBoost risk stratification model.

Gold tables produced:
  - member_raf_scores:  one row per member with final RAF score, risk tier, cost
  - utilization_summary: IP admits / ED visits / PMPM by member-period
  - monthly_trends:      cohort-level monthly aggregate for DiD analysis
  - risk_feature_store:  ML-ready feature matrix for model training
"""

from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config.pipeline_config import CONFIG, GoldConfig
from src.utils.spark_utils import (
    add_audit_columns, get_logger, write_delta, get_spark,
)

logger = get_logger(__name__)


class GoldAggregator:
    """
    Builds all Gold-layer analytics tables from Silver inputs.

    Pipeline:
      1. Member-level RAF score calculation
      2. Risk tier assignment
      3. Utilization summary (IP, ED, PMPM)
      4. Monthly cohort trends (for DiD-style analysis)
      5. ML feature store construction

    Usage:
        gold = GoldAggregator(spark)
        results = gold.run()
    """

    def __init__(
        self,
        spark:  Optional[SparkSession] = None,
        config: Optional[GoldConfig] = None,
    ):
        self.spark  = spark or get_spark()
        self.config = config or CONFIG.gold

    # ── Stage 1: RAF Score Aggregation ─────────────────────────────────

    def _build_raf_scores(self) -> DataFrame:
        """
        Compute member-level RAF scores by joining member demographics
        with their HCC burden from Silver claims.

        RAF = demographic_raf + sum(unique HCC coefficients) + interaction terms
        """
        logger.info("Building member RAF scores")

        # Aggregate HCC burden per member (across all claim dates)
        hcc_per_member = (
            self.spark.table(self.config.hcc_mapped_table)
            .groupBy("bene_id")
            .agg(
                F.max("hcc_burden_count").alias("max_hcc_burden"),
                F.max("hcc_raf_total").alias("max_hcc_raf"),
                F.max("hcc_interaction_total").alias("max_interaction_raf"),
                F.max("has_chf").alias("has_chf"),
                F.max("has_afib").alias("has_afib"),
                F.max("has_diabetes").alias("has_diabetes"),
                F.max("has_ckd").alias("has_ckd"),
                F.max("has_cancer").alias("has_cancer"),
                F.max("has_copd").alias("has_copd"),
                F.max("has_depression").alias("has_depression"),
                F.max("has_metastatic").alias("has_metastatic"),
                F.sum("claim_amount").alias("total_claim_amount"),
                F.count("claim_id").alias("total_claims"),
                F.sum(F.col("is_inpatient").cast("int")).alias("ip_admit_count"),
                F.sum(F.col("is_ed_visit").cast("int")).alias("ed_visit_count"),
                F.countDistinct(
                    F.when(F.col("period") == "pre",  F.col("service_month"))
                ).alias("pre_months"),
                F.countDistinct(
                    F.when(F.col("period") == "post", F.col("service_month"))
                ).alias("post_months"),
                F.sum(
                    F.when(F.col("period") == "pre",  F.col("claim_amount"))
                ).alias("pre_total_cost"),
                F.sum(
                    F.when(F.col("period") == "post", F.col("claim_amount"))
                ).alias("post_total_cost"),
            )
        )

        # Join member demographics
        members = self.spark.table(self.config.member_profile_table)

        raf_df = members.join(hcc_per_member, on="bene_id", how="left")

        # Fill nulls for members with no claims
        for col in ["max_hcc_burden", "max_hcc_raf", "max_interaction_raf",
                    "total_claim_amount", "total_claims", "ip_admit_count", "ed_visit_count"]:
            raf_df = raf_df.fillna({col: 0})

        # Final RAF score
        raf_df = (
            raf_df
            .withColumn("raf_score",
                F.col("demographic_raf") +
                F.coalesce(F.col("max_hcc_raf"),         F.lit(0.0)) +
                F.coalesce(F.col("max_interaction_raf"),  F.lit(0.0))
            )
            # Risk tier assignment
            .withColumn("risk_tier", F.when(
                F.col("raf_score") >= 2.0, "high"
            ).when(
                F.col("raf_score") >= 1.2, "moderate"
            ).otherwise("low"))
            # Annual cost estimate: RAF × benchmark PMPM × 12
            .withColumn("estimated_annual_cost",
                F.col("raf_score") * self.config.benchmark_pmpm * 12
            )
            # PMPM from actual claims
            .withColumn("actual_pmpm",
                F.col("total_claim_amount") / F.greatest(
                    F.col("pre_months") + F.col("post_months"), F.lit(1)
                )
            )
            # Pre / post PMPM for DiD
            .withColumn("pre_pmpm",
                F.col("pre_total_cost") / F.greatest(F.col("pre_months"), F.lit(1))
            )
            .withColumn("post_pmpm",
                F.col("post_total_cost") / F.greatest(F.col("post_months"), F.lit(1))
            )
        )

        logger.info(f"RAF scores computed for {raf_df.count():,} members")
        return raf_df

    # ── Stage 2: Utilization Summary ────────────────────────────────────

    def _build_utilization(self) -> DataFrame:
        """
        Build a member × period utilization summary.
        Includes IP admits, ED visits, PMPM cost, and utilization rates.
        """
        logger.info("Building utilization summary")

        util = (
            self.spark.table(self.config.clean_claims_table)
            .filter(F.col("_quality_pass") == True)
            .groupBy("bene_id", "period", "intervention_arm")
            .agg(
                F.countDistinct("service_date").alias("service_days"),
                F.count("claim_id").alias("total_claims"),
                F.sum("claim_amount").alias("total_cost"),
                F.sum("allowed_amount").alias("total_allowed"),
                F.sum("paid_amount").alias("total_paid"),
                F.sum(F.col("is_inpatient").cast("int")).alias("ip_admits"),
                F.sum(F.col("is_ed_visit").cast("int")).alias("ed_visits"),
                F.sum(F.col("is_specialist").cast("int")).alias("specialist_visits"),
                F.sum(F.col("is_primary_care").cast("int")).alias("primary_care_visits"),
                F.countDistinct("service_year", "service_month").alias("n_months"),
            )
            .withColumn("pmpm_cost",
                F.col("total_cost") / F.greatest(F.col("n_months"), F.lit(1))
            )
            .withColumn("ip_rate_per_1000",
                F.col("ip_admits") * 1000 / F.greatest(F.col("n_months"), F.lit(1)) / 12
            )
            .withColumn("ed_rate_per_1000",
                F.col("ed_visits") * 1000 / F.greatest(F.col("n_months"), F.lit(1)) / 12
            )
        )

        logger.info(f"Utilization summary: {util.count():,} member-period rows")
        return util

    # ── Stage 3: Monthly Cohort Trends ──────────────────────────────────

    def _build_monthly_trends(self) -> DataFrame:
        """
        Cohort-level monthly aggregates for DiD-style analysis.
        One row per (year, month, intervention_arm).
        """
        logger.info("Building monthly trend table")

        trends = (
            self.spark.table(self.config.clean_claims_table)
            .filter(F.col("_quality_pass") == True)
            .groupBy("service_year", "service_month", "intervention_arm")
            .agg(
                F.countDistinct("bene_id").alias("n_members"),
                F.sum("claim_amount").alias("total_cost"),
                F.sum(F.col("is_inpatient").cast("int")).alias("ip_admits"),
                F.sum(F.col("is_ed_visit").cast("int")).alias("ed_visits"),
            )
            .withColumn("pmpm_cost",
                F.col("total_cost") / F.greatest(F.col("n_members"), F.lit(1))
            )
            .withColumn("ip_rate_per_1000",
                F.col("ip_admits") * 1000 / F.greatest(F.col("n_members"), F.lit(1))
            )
            .withColumn("ed_rate_per_1000",
                F.col("ed_visits") * 1000 / F.greatest(F.col("n_members"), F.lit(1))
            )
            .withColumn("period",
                F.when(F.col("service_year") >= 2023, "post").otherwise("pre")
            )
            # 3-month rolling average PMPM
            .withColumn("pmpm_rolling_3m",
                F.avg("pmpm_cost").over(
                    Window
                    .partitionBy("intervention_arm")
                    .orderBy("service_year", "service_month")
                    .rowsBetween(-2, 0)
                )
            )
            .orderBy("service_year", "service_month", "intervention_arm")
        )

        logger.info(f"Monthly trends: {trends.count():,} rows")
        return trends

    # ── Stage 4: ML Feature Store ────────────────────────────────────────

    def _build_feature_store(self, raf_df: DataFrame) -> DataFrame:
        """
        Construct an ML-ready feature matrix by joining RAF scores with
        utilization patterns. This is the direct input to XGBoost training.

        Feature groups:
          - Demographic:    age, sex, dual_eligible, state
          - HCC burden:     hcc_burden_count, condition flags
          - RAF components: demographic_raf, hcc_raf, interaction_raf
          - Utilization:    pre/post PMPM, IP rate, ED rate
          - Labels:         risk_tier (classification), annual cost (regression)
        """
        logger.info("Building ML feature store")

        util = (
            self.spark.table(self.config.clean_claims_table)
            .filter(
                (F.col("_quality_pass") == True) &
                (F.col("period") == "pre")  # use pre-period only for training features
            )
            .groupBy("bene_id")
            .agg(
                F.sum(F.col("is_inpatient").cast("int")).alias("pre_ip_admits"),
                F.sum(F.col("is_ed_visit").cast("int")).alias("pre_ed_visits"),
                F.sum(F.col("is_specialist").cast("int")).alias("pre_specialist_visits"),
                F.sum(F.col("is_primary_care").cast("int")).alias("pre_primary_visits"),
                F.countDistinct("service_year", "service_month").alias("pre_n_months"),
                F.sum("claim_amount").alias("pre_total_cost"),
                F.avg("claim_amount").alias("pre_avg_claim"),
                F.max("claim_amount").alias("pre_max_claim"),
            )
            .withColumn("pre_pmpm",
                F.col("pre_total_cost") / F.greatest(F.col("pre_n_months"), F.lit(1))
            )
            .withColumn("pre_ip_rate",
                F.col("pre_ip_admits") / F.greatest(F.col("pre_n_months"), F.lit(1))
            )
            .withColumn("pre_ed_rate",
                F.col("pre_ed_visits") / F.greatest(F.col("pre_n_months"), F.lit(1))
            )
        )

        # Join RAF scores with utilization features
        features = (
            raf_df.join(util, on="bene_id", how="left")
            .select(
                # Key
                "bene_id",
                # Demographic features
                "age", "sex", "dual_eligible", "state",
                "demographic_raf", "age_bracket",
                # HCC / clinical features
                "max_hcc_burden", "max_hcc_raf", "max_interaction_raf",
                "has_chf", "has_afib", "has_diabetes", "has_ckd",
                "has_cancer", "has_copd", "has_depression", "has_metastatic",
                # Utilization features (pre-period)
                "pre_ip_admits", "pre_ed_visits", "pre_specialist_visits",
                "pre_primary_visits", "pre_n_months", "pre_total_cost",
                "pre_avg_claim", "pre_max_claim", "pre_pmpm",
                "pre_ip_rate", "pre_ed_rate",
                # RAF and costs
                "raf_score", "estimated_annual_cost", "actual_pmpm",
                # Label columns
                "risk_tier",
                "intervention_arm",
            )
            .fillna(0)
        )

        # One-hot encode sex (male = 1)
        features = features.withColumn("sex_male", (F.col("sex") == "M").cast("int"))

        # Log-transform skewed cost features
        features = (
            features
            .withColumn("log_pre_pmpm",        F.log1p(F.col("pre_pmpm")))
            .withColumn("log_estimated_cost",  F.log1p(F.col("estimated_annual_cost")))
            .withColumn("log_pre_total_cost",  F.log1p(F.col("pre_total_cost")))
        )

        logger.info(f"Feature store built: {features.count():,} members, {len(features.columns)} features")
        return features

    # ── Main runner ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full Gold aggregation pipeline.

        Returns:
            dict with row counts, table names, and summary statistics.
        """
        logger.info("Gold aggregation starting")

        from config.pipeline_config import CATALOG, GOLD_SCHEMA, ML_SCHEMA
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{ML_SCHEMA}")

        # RAF scores
        raf_df    = self._build_raf_scores()
        raf_df    = add_audit_columns(raf_df, "gold")
        raf_count = write_delta(
            raf_df, self.config.raf_scores_table,
            mode="overwrite", optimize_after=True,
            zorder_cols=["bene_id", "risk_tier"], spark=self.spark
        )

        # Utilization
        util_df    = self._build_utilization()
        util_df    = add_audit_columns(util_df, "gold")
        util_count = write_delta(
            util_df, self.config.utilization_table,
            mode="overwrite", partition_by=["period"], spark=self.spark
        )

        # Monthly trends
        trends_df    = self._build_monthly_trends()
        trends_df    = add_audit_columns(trends_df, "gold")
        trends_count = write_delta(
            trends_df, self.config.monthly_trends_table,
            mode="overwrite", spark=self.spark
        )

        # Feature store
        feat_df    = self._build_feature_store(raf_df)
        feat_df    = add_audit_columns(feat_df, "gold")
        feat_count = write_delta(
            feat_df, self.config.feature_store_table,
            mode="overwrite", optimize_after=True,
            zorder_cols=["risk_tier"], spark=self.spark
        )

        # Summary stats for logging
        tier_counts = (
            raf_df.groupBy("risk_tier")
            .count()
            .collect()
        )
        tier_summary = {row["risk_tier"]: row["count"] for row in tier_counts}
        mean_raf = raf_df.agg(F.mean("raf_score")).collect()[0][0]

        result = {
            "raf_rows":     raf_count,
            "util_rows":    util_count,
            "trends_rows":  trends_count,
            "feature_rows": feat_count,
            "mean_raf":     round(float(mean_raf), 3),
            "tier_counts":  tier_summary,
            "tables": {
                "raf_scores":     self.config.raf_scores_table,
                "utilization":    self.config.utilization_table,
                "monthly_trends": self.config.monthly_trends_table,
                "feature_store":  self.config.feature_store_table,
            }
        }

        logger.info(
            f"Gold complete — RAF rows: {raf_count:,} | "
            f"mean RAF: {mean_raf:.3f} | tiers: {tier_summary}"
        )
        return result


def run_gold_aggregation(spark: Optional[SparkSession] = None) -> dict:
    """One-line entry point for Databricks notebooks."""
    return GoldAggregator(spark=spark).run()
