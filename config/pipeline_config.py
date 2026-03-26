"""
config/pipeline_config.py
─────────────────────────
Central configuration for the Medicare Claims Lakehouse Pipeline.
All layer paths, catalog names, model parameters, and thresholds
are defined here so no magic strings appear elsewhere in the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Unity Catalog ──────────────────────────────────────────────────────────

CATALOG        = "medicare_lakehouse"
BRONZE_SCHEMA  = "bronze"
SILVER_SCHEMA  = "silver"
GOLD_SCHEMA    = "gold"
ML_SCHEMA      = "ml_features"


# ── Delta Lake Paths (Community Edition falls back to DBFS) ────────────────

BRONZE_BASE  = f"/mnt/{CATALOG}/bronze"
SILVER_BASE  = f"/mnt/{CATALOG}/silver"
GOLD_BASE    = f"/mnt/{CATALOG}/gold"
ML_BASE      = f"/mnt/{CATALOG}/ml_features"
RAW_LANDING  = f"/mnt/{CATALOG}/raw/claims"


# ── Table Names ────────────────────────────────────────────────────────────

class Tables:
    # Bronze
    RAW_CLAIMS       = f"{CATALOG}.{BRONZE_SCHEMA}.raw_claims"
    RAW_MEMBERS      = f"{CATALOG}.{BRONZE_SCHEMA}.raw_members"

    # Silver
    CLEAN_CLAIMS     = f"{CATALOG}.{SILVER_SCHEMA}.clean_claims"
    HCC_MAPPED       = f"{CATALOG}.{SILVER_SCHEMA}.hcc_mapped_claims"
    MEMBER_PROFILE   = f"{CATALOG}.{SILVER_SCHEMA}.member_profile"

    # Gold
    RAF_SCORES       = f"{CATALOG}.{GOLD_SCHEMA}.member_raf_scores"
    UTILIZATION      = f"{CATALOG}.{GOLD_SCHEMA}.utilization_summary"
    MONTHLY_TRENDS   = f"{CATALOG}.{GOLD_SCHEMA}.monthly_trends"
    FEATURE_STORE    = f"{CATALOG}.{ML_SCHEMA}.risk_feature_store"


# ── Data Generator ─────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    n_members:            int   = 10_000
    n_months:             int   = 24          # 2 years of synthetic history
    intervention_effect:  float = -420.0      # PMPM cost reduction for treated group
    seed:                 int   = 42
    output_path:          str   = RAW_LANDING


# ── Bronze Layer ───────────────────────────────────────────────────────────

@dataclass
class BronzeConfig:
    raw_path:         str  = RAW_LANDING
    claims_table:     str  = Tables.RAW_CLAIMS
    members_table:    str  = Tables.RAW_MEMBERS
    partition_cols:   List[str] = field(default_factory=lambda: ["claim_year", "claim_month"])
    enable_schema_enforcement: bool = True
    checkpoint_path:  str  = f"{BRONZE_BASE}/_checkpoints"


# ── Silver Layer ───────────────────────────────────────────────────────────

@dataclass
class SilverConfig:
    # Input
    bronze_claims_table:  str  = Tables.RAW_CLAIMS
    bronze_members_table: str  = Tables.RAW_MEMBERS

    # Output
    clean_claims_table:   str  = Tables.CLEAN_CLAIMS
    hcc_mapped_table:     str  = Tables.HCC_MAPPED
    member_profile_table: str  = Tables.MEMBER_PROFILE

    # Quality thresholds
    max_claim_amount:     float = 500_000.0
    min_claim_amount:     float = 0.01
    max_age:              int   = 120
    min_age:              int   = 0

    # Dedup key
    dedup_keys: List[str] = field(default_factory=lambda: [
        "claim_id", "bene_id", "service_date"
    ])


# ── Gold Layer ─────────────────────────────────────────────────────────────

@dataclass
class GoldConfig:
    # Input
    hcc_mapped_table:    str = Tables.HCC_MAPPED
    member_profile_table: str = Tables.MEMBER_PROFILE
    clean_claims_table:  str = Tables.CLEAN_CLAIMS

    # Output
    raf_scores_table:    str = Tables.RAF_SCORES
    utilization_table:   str = Tables.UTILIZATION
    monthly_trends_table: str = Tables.MONTHLY_TRENDS
    feature_store_table: str = Tables.FEATURE_STORE

    # RAF benchmark
    benchmark_pmpm:      float = 9_800.0 / 12   # Monthly benchmark
    msr_threshold:       float = 0.02            # MSSP minimum savings rate
    sharing_rate:        float = 0.50            # MSSP sharing rate


# ── ML / MLflow ────────────────────────────────────────────────────────────

@dataclass
class MLConfig:
    experiment_name:  str   = "/Shared/medicare_raf_risk_stratification"
    model_name:       str   = "medicare_raf_risk_model"
    feature_table:    str   = Tables.FEATURE_STORE
    target_col:       str   = "risk_tier"
    cost_target_col:  str   = "estimated_annual_cost"
    test_size:        float = 0.20
    random_state:     int   = 42

    # XGBoost hyperparameters
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators":      300,
        "max_depth":         6,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  3,
        "random_state":      42,
        "use_label_encoder": False,
        "eval_metric":       "mlogloss",
    })

    # Drift thresholds (PSI)
    psi_warning_threshold:  float = 0.10
    psi_critical_threshold: float = 0.25

    # Model aliases
    production_alias: str = "Production"
    staging_alias:    str = "Staging"


# ── Governance ─────────────────────────────────────────────────────────────

@dataclass
class GovernanceConfig:
    catalog:         str = CATALOG
    schemas:         List[str] = field(default_factory=lambda: [
        BRONZE_SCHEMA, SILVER_SCHEMA, GOLD_SCHEMA, ML_SCHEMA
    ])
    analyst_group:   str = "analysts@acohealth.org"
    engineer_group:  str = "engineers@acohealth.org"
    scientist_group: str = "datascientists@acohealth.org"
    auditor_group:   str = "auditors@acohealth.org"


# ── Pipeline-level convenience object ─────────────────────────────────────

@dataclass
class PipelineConfig:
    generator:   GeneratorConfig   = field(default_factory=GeneratorConfig)
    bronze:      BronzeConfig      = field(default_factory=BronzeConfig)
    silver:      SilverConfig      = field(default_factory=SilverConfig)
    gold:        GoldConfig        = field(default_factory=GoldConfig)
    ml:          MLConfig          = field(default_factory=MLConfig)
    governance:  GovernanceConfig  = field(default_factory=GovernanceConfig)


# Default singleton — import and use directly
CONFIG = PipelineConfig()
