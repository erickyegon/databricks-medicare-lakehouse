"""
src/governance/unity_catalog_setup.py
──────────────────────────────────────
Unity Catalog governance: catalog/schema creation, GRANT/REVOKE access
controls, data lineage tagging, and audit-ready table properties.

Access model (ACO data environment):
  ┌─────────────────┬──────────────────────────────────────────┐
  │ Group           │ Permissions                              │
  ├─────────────────┼──────────────────────────────────────────┤
  │ analysts        │ SELECT on gold.*, ml_features.*          │
  │ engineers       │ ALL on bronze.*, silver.*                │
  │ data scientists │ ALL on ml_features.*, SELECT on silver.* │
  │ auditors        │ SELECT on ALL schemas (read-only audit)  │
  └─────────────────┴──────────────────────────────────────────┘

Note: Unity Catalog commands require Databricks Runtime 11.3+
and a Unity Catalog-enabled workspace.
"""

from typing import List, Optional
from pyspark.sql import SparkSession

from config.pipeline_config import CONFIG, GovernanceConfig, Tables
from src.utils.spark_utils import get_logger, get_spark

logger = get_logger(__name__)


class UnityCatalogGovernance:
    """
    Manages Unity Catalog structure and access controls for the
    Medicare Claims Lakehouse.

    Usage:
        gov = UnityCatalogGovernance(spark)
        gov.setup_all()
    """

    def __init__(
        self,
        spark:  Optional[SparkSession] = None,
        config: Optional[GovernanceConfig] = None,
    ):
        self.spark  = spark or get_spark()
        self.config = config or CONFIG.governance

    # ── Catalog and schema setup ────────────────────────────────────────

    def create_catalog(self) -> None:
        """Create the top-level catalog with HIPAA-compliant comment."""
        self.spark.sql(f"""
            CREATE CATALOG IF NOT EXISTS {self.config.catalog}
            COMMENT 'Medicare Claims Lakehouse — HIPAA-compliant ACO analytics platform.
                     Contains synthetic CMS claims data. PHI handling: de-identified synthetic only.'
        """)
        logger.info(f"Catalog ready: {self.config.catalog}")

    def create_schemas(self) -> None:
        """Create bronze / silver / gold / ml_features schemas."""
        schema_comments = {
            "bronze":      "Raw ingestion layer — append-only, never modified",
            "silver":      "Cleaned and HCC-enriched claims — quality-validated",
            "gold":        "Analytics-ready aggregates — RAF scores, utilization, trends",
            "ml_features": "ML feature store — XGBoost training inputs and predictions",
        }
        self.spark.sql(f"USE CATALOG {self.config.catalog}")
        for schema in self.config.schemas:
            comment = schema_comments.get(schema, "")
            self.spark.sql(f"""
                CREATE SCHEMA IF NOT EXISTS {schema}
                COMMENT '{comment}'
            """)
            logger.info(f"Schema ready: {self.config.catalog}.{schema}")

    # ── Access controls ─────────────────────────────────────────────────

    def grant_analyst_access(self) -> None:
        """Analysts: read Gold and ML features — no raw data access."""
        group = self.config.analyst_group
        stmts = [
            f"GRANT USE CATALOG ON CATALOG {self.config.catalog} TO `{group}`",
            f"GRANT USE SCHEMA ON SCHEMA {self.config.catalog}.gold TO `{group}`",
            f"GRANT USE SCHEMA ON SCHEMA {self.config.catalog}.ml_features TO `{group}`",
            f"GRANT SELECT ON TABLE {Tables.RAF_SCORES} TO `{group}`",
            f"GRANT SELECT ON TABLE {Tables.UTILIZATION} TO `{group}`",
            f"GRANT SELECT ON TABLE {Tables.MONTHLY_TRENDS} TO `{group}`",
            f"GRANT SELECT ON TABLE {Tables.FEATURE_STORE} TO `{group}`",
        ]
        self._execute_grants(stmts, f"analyst access → {group}")

    def grant_engineer_access(self) -> None:
        """Engineers: full access to bronze and silver for pipeline management."""
        group = self.config.engineer_group
        stmts = [
            f"GRANT USE CATALOG ON CATALOG {self.config.catalog} TO `{group}`",
            f"GRANT ALL PRIVILEGES ON SCHEMA {self.config.catalog}.bronze TO `{group}`",
            f"GRANT ALL PRIVILEGES ON SCHEMA {self.config.catalog}.silver TO `{group}`",
            # Read-only on gold — engineers don't overwrite business aggregates
            f"GRANT USE SCHEMA ON SCHEMA {self.config.catalog}.gold TO `{group}`",
            f"GRANT SELECT ON SCHEMA {self.config.catalog}.gold TO `{group}`",
        ]
        self._execute_grants(stmts, f"engineer access → {group}")

    def grant_data_scientist_access(self) -> None:
        """Data scientists: full ML features, read silver for feature development."""
        group = self.config.scientist_group
        stmts = [
            f"GRANT USE CATALOG ON CATALOG {self.config.catalog} TO `{group}`",
            f"GRANT ALL PRIVILEGES ON SCHEMA {self.config.catalog}.ml_features TO `{group}`",
            f"GRANT USE SCHEMA ON SCHEMA {self.config.catalog}.silver TO `{group}`",
            f"GRANT SELECT ON SCHEMA {self.config.catalog}.silver TO `{group}`",
            f"GRANT USE SCHEMA ON SCHEMA {self.config.catalog}.gold TO `{group}`",
            f"GRANT SELECT ON SCHEMA {self.config.catalog}.gold TO `{group}`",
        ]
        self._execute_grants(stmts, f"data scientist access → {group}")

    def grant_auditor_access(self) -> None:
        """Auditors: read-only across all schemas for compliance review."""
        group = self.config.auditor_group
        stmts = [
            f"GRANT USE CATALOG ON CATALOG {self.config.catalog} TO `{group}`",
        ]
        for schema in self.config.schemas:
            stmts += [
                f"GRANT USE SCHEMA ON SCHEMA {self.config.catalog}.{schema} TO `{group}`",
                f"GRANT SELECT ON SCHEMA {self.config.catalog}.{schema} TO `{group}`",
            ]
        self._execute_grants(stmts, f"auditor access → {group}")

    def _execute_grants(self, statements: List[str], label: str) -> None:
        """Execute a list of GRANT statements, logging each one."""
        for stmt in statements:
            try:
                self.spark.sql(stmt)
                logger.info(f"GRANT OK [{label}]: {stmt[:80]}...")
            except Exception as e:
                # In Community Edition, GRANT is not supported — log and continue
                logger.warning(f"GRANT skipped (may need Unity Catalog workspace): {e}")

    # ── Table properties & lineage tags ─────────────────────────────────

    def set_table_properties(self) -> None:
        """
        Add table-level properties for lineage tracking and HIPAA compliance.
        These appear in the Unity Catalog UI and data lineage graph.
        """
        table_props = {
            Tables.RAW_CLAIMS: {
                "data_classification": "HIPAA_deidentified",
                "pipeline_layer":      "bronze",
                "source":              "cms_synthetic_generator",
                "retention_days":      "2555",   # 7 years per HIPAA
                "owner":               "data_engineering",
            },
            Tables.CLEAN_CLAIMS: {
                "data_classification": "HIPAA_deidentified",
                "pipeline_layer":      "silver",
                "derived_from":        Tables.RAW_CLAIMS,
                "quality_validated":   "true",
            },
            Tables.HCC_MAPPED: {
                "data_classification": "HIPAA_deidentified",
                "pipeline_layer":      "silver",
                "hcc_version":         "v28",
                "derived_from":        Tables.CLEAN_CLAIMS,
            },
            Tables.RAF_SCORES: {
                "data_classification": "HIPAA_deidentified",
                "pipeline_layer":      "gold",
                "business_metric":     "raf_score",
                "cms_model":           "HCC_v28",
                "derived_from":        Tables.HCC_MAPPED,
            },
            Tables.FEATURE_STORE: {
                "pipeline_layer":      "ml_features",
                "ml_model":            "xgboost_risk_stratification",
                "feature_count":       str(len(__import__(
                    "src.ml.risk_model", fromlist=["CLASSIFICATION_FEATURES"]
                ).CLASSIFICATION_FEATURES)),
                "derived_from":        Tables.RAF_SCORES,
            },
        }

        for table, props in table_props.items():
            prop_str = ", ".join([f"'{k}' = '{v}'" for k, v in props.items()])
            try:
                self.spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES ({prop_str})")
                logger.info(f"Table properties set: {table}")
            except Exception as e:
                logger.warning(f"Table properties skipped for {table}: {e}")

    # ── Controlled views ─────────────────────────────────────────────────

    def create_analyst_views(self) -> None:
        """
        Create analyst-facing views in Gold that surface pre-aggregated,
        approved metrics without exposing member-level identifiers.
        """
        views = {
            f"{self.config.catalog}.gold.v_high_risk_summary": f"""
                SELECT
                    risk_tier,
                    COUNT(*)                    AS n_members,
                    ROUND(AVG(raf_score), 3)    AS mean_raf,
                    ROUND(AVG(actual_pmpm), 2)  AS mean_pmpm,
                    SUM(ip_admit_count)         AS total_ip_admits,
                    SUM(ed_visit_count)         AS total_ed_visits
                FROM {Tables.RAF_SCORES}
                GROUP BY risk_tier
            """,
            f"{self.config.catalog}.gold.v_cohort_trends": f"""
                SELECT
                    service_year,
                    service_month,
                    intervention_arm,
                    n_members,
                    ROUND(pmpm_cost, 2)         AS pmpm_cost,
                    ROUND(pmpm_rolling_3m, 2)   AS pmpm_3m_avg,
                    ip_rate_per_1000,
                    ed_rate_per_1000,
                    period
                FROM {Tables.MONTHLY_TRENDS}
                ORDER BY service_year, service_month, intervention_arm
            """,
            f"{self.config.catalog}.gold.v_top_hcc_conditions": f"""
                SELECT
                    primary_hcc_desc            AS condition,
                    COUNT(DISTINCT bene_id)     AS n_members,
                    ROUND(AVG(hcc_raf_total), 3)AS mean_hcc_raf,
                    ROUND(SUM(claim_amount), 0) AS total_claims_usd
                FROM {Tables.HCC_MAPPED}
                WHERE primary_hcc_desc IS NOT NULL
                  AND _quality_pass = true
                GROUP BY primary_hcc_desc
                ORDER BY n_members DESC
                LIMIT 20
            """,
        }

        for view_name, view_sql in views.items():
            try:
                self.spark.sql(f"""
                    CREATE OR REPLACE VIEW {view_name}
                    COMMENT 'Analyst view — no member-level identifiers'
                    AS {view_sql}
                """)
                logger.info(f"View created: {view_name}")
            except Exception as e:
                logger.warning(f"View creation skipped for {view_name}: {e}")

    # ── Full setup orchestration ─────────────────────────────────────────

    def setup_all(self) -> dict:
        """
        Run the complete governance setup in order:
          1. Catalog + schemas
          2. Access controls (all groups)
          3. Table properties + lineage tags
          4. Analyst views
        """
        logger.info("Unity Catalog governance setup starting")

        self.create_catalog()
        self.create_schemas()

        self.grant_analyst_access()
        self.grant_engineer_access()
        self.grant_data_scientist_access()
        self.grant_auditor_access()

        self.set_table_properties()
        self.create_analyst_views()

        result = {
            "catalog":  self.config.catalog,
            "schemas":  self.config.schemas,
            "groups": {
                "analysts":        self.config.analyst_group,
                "engineers":       self.config.engineer_group,
                "data_scientists": self.config.scientist_group,
                "auditors":        self.config.auditor_group,
            },
            "views_created": [
                "v_high_risk_summary",
                "v_cohort_trends",
                "v_top_hcc_conditions",
            ]
        }

        logger.info(f"Governance setup complete: {result}")
        return result


def run_governance_setup(spark: Optional[SparkSession] = None) -> dict:
    """One-line entry point for Databricks notebooks."""
    return UnityCatalogGovernance(spark=spark).setup_all()
