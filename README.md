# 🏥 Medicare Claims Lakehouse Pipeline

> **A production-grade, end-to-end Databricks Lakehouse** for Medicare Advantage risk analytics — architected from first principles with CMS HCC v28 risk adjustment, medallion ETL, XGBoost risk stratification, SHAP explainability, Unity Catalog governance, and automated Workflow orchestration. Built to reflect the actual data engineering infrastructure of a Medicare ACO operating under MSSP.

<br>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Databricks Runtime 13+](https://img.shields.io/badge/Databricks-Runtime%2013.3%20LTS-FF3621?logo=databricks&logoColor=white)](https://docs.databricks.com/runtime/)
[![Delta Lake 3.x](https://img.shields.io/badge/Delta%20Lake-3.x-003366)](https://delta.io/)
[![MLflow 2.x](https://img.shields.io/badge/MLflow-2.x-0194E2)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-337AB7)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43%2B-FF6B6B)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-55%2B%20assertions-22C55E)](tests/)

<br>

---

## 📋 Table of Contents

- [Why This Project Exists](#-why-this-project-exists)
- [Business Context — Medicare ACO Analytics](#-business-context--medicare-aco-analytics)
- [Pipeline Architecture](#-pipeline-architecture)
- [Phase 1 — Data Engineering Foundation](#-phase-1--data-engineering-foundation)
- [Phase 2 — Clinical Enrichment (Silver)](#-phase-2--clinical-enrichment-silver)
- [Phase 3 — Analytics-Ready Gold Layer](#-phase-3--analytics-ready-gold-layer)
- [Phase 4 — ML Risk Stratification](#-phase-4--ml-risk-stratification)
- [Phase 5 — Governance and Orchestration](#-phase-5--governance-and-orchestration)
- [Key Technical Decisions](#-key-technical-decisions)
- [Performance Benchmarks](#-performance-benchmarks)
- [Data Dictionary](#-data-dictionary)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Troubleshooting](#-troubleshooting)
- [Related Portfolio Projects](#-related-portfolio-projects)

---

## 🎯 Why This Project Exists

Most data engineering tutorials stop at loading a CSV into a Delta table and calling it a "lakehouse." This project does not.

This pipeline was built to answer a specific question: **what does the actual data infrastructure of a Medicare Advantage ACO look like when built properly on Databricks?** The answer involves:

- **Claims data that is never modified after ingestion** — Bronze is append-only by design, not convention
- **Clinical enrichment that follows CMS HCC v28 rules** — not generic feature engineering, but the actual regulatory framework that determines how Medicare pays health plans
- **Causal analysis embedded in the data model** — monthly cohort trends are structured for Difference-in-Differences from the moment they land in Gold
- **A model that explains itself** — SHAP waterfall plots per member, because clinicians will not act on a black-box score
- **Governance that mirrors a real ACO** — analysts see Gold only, engineers own Bronze/Silver, auditors get read-only access across all schemas
- **A pipeline that runs itself** — Databricks Workflows orchestrates the full DAG on a daily schedule with failure alerting

The companion project [medicare-raf-prototypes](https://github.com/erickyegon/medicare-raf-prototypes) demonstrated that a care management intervention reduced annual cost by **ATT = −$391/member/year (p < 0.0001)**. This project builds the Lakehouse infrastructure that would run that analysis continuously, in production, at enterprise scale.

---

## 🏛️ Business Context — Medicare ACO Analytics

### What is a Medicare ACO?

An **Accountable Care Organization (ACO)** is a group of healthcare providers that jointly takes responsibility for the cost and quality of care for a defined population of Medicare beneficiaries. Under the **Medicare Shared Savings Programme (MSSP)**, the ACO earns a share of any savings it generates relative to a CMS-set expenditure benchmark — provided savings exceed the **Minimum Savings Rate (MSR)** threshold.

The economics depend entirely on accurate **Risk Adjustment Factor (RAF) scores**. RAF is how CMS adjusts capitation payments to reflect each member's expected cost burden. A member with RAF = 2.0 is expected to cost twice the Medicare average. Miscalculated RAF = miscalculated revenue.

### What this pipeline computes

| Analytical Output | Business Purpose | CMS Regulatory Connection |
|-------------------|------------------|---------------------------|
| Member RAF scores | Capitation payment calibration | CMS HCC v28 risk adjustment model |
| Risk tier assignments | Care management programme targeting | High-risk = intensive outreach priority |
| PMPM utilization trends | Financial performance monitoring | Benchmark vs actual for MSSP filing |
| Monthly DiD cohort trends | Causal intervention evaluation | Programme impact quantification |
| ML risk feature store | Predictive targeting before cost events | Prospective risk identification |
| SHAP per-member explanations | Clinical adoption of AI scores | IRB-compliant model transparency |

### Shared Savings Mechanics (embedded in Gold layer)

```
Benchmark PMPM = CMS-set expected cost per member per month
Actual PMPM    = Observed PMPM from claims (computed by this pipeline)
Gross Savings  = (Benchmark − Actual) × N attributed lives × 12

If Gross Savings / (Benchmark × N × 12) > MSR (2%):
    ACO Earned = Gross Savings × Sharing Rate (50%)
```

At **ATT = −$391/member/year** across a 50,000-member plan:
- Gross savings: ~$19.5M
- Savings rate: ~4.0% (exceeds 2% MSR)
- ACO earned at 50% sharing: **~$9.75M**

---

## 🏗️ Pipeline Architecture

### End-to-End Data Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║  DATA SOURCE                                                         ║
║  CMS Synthetic Claims Generator                                      ║
║  • 10,000 beneficiaries  •  24 months  •  ~250K claim lines         ║
║  • ICD-10 codes drawn from 27-condition weighted HCC catalogue       ║
║  • Control / intervention arm assignment for causal analysis         ║
╚═══════════════════════════╦══════════════════════════════════════════╝
                            ║
                            ▼
╔══════════════════════════════════════════════════════════════════════╗
║  BRONZE LAYER  (append-only, immutable, partition-optimised)        ║
║                                                                      ║
║  bronze.raw_claims                bronze.raw_members                ║
║  ├── Schema enforcement           ├── Full snapshot per batch        ║
║  ├── Idempotent ingestion         ├── Bronze null flags              ║
║  ├── _batch_id for lineage        └── Audit columns injected         ║
║  ├── Partition: claim_year / claim_month                             ║
║  ├── ZORDER BY bene_id                                               ║
║  └── _bronze_null_flag column                                        ║
╚═══════════════════════════╦══════════════════════════════════════════╝
                            ║ APPEND (idempotent, batch-keyed)
                            ▼
╔══════════════════════════════════════════════════════════════════════╗
║  SILVER LAYER  (clean, enrich, validate)                            ║
║                                                                      ║
║  silver.clean_claims              silver.hcc_mapped_claims           ║
║  ├── Deduplication                ├── ICD-10 → HCC v28 Spark UDFs   ║
║  │   (row_number on dedup keys,   ├── RAF coefficient aggregation    ║
║  │    keeps max paid_amount)      ├── CHF×AFib interaction (+0.175)  ║
║  ├── 9 quality flags              ├── CKD×Diabetes interaction       ║
║  │   + _quality_pass column       ├── 8 boolean condition flags      ║
║  ├── Date feature extraction      └── hcc_burden_count               ║
║  ├── Service type categorisation                                     ║
║  └── Pre / post period flag       silver.member_profile              ║
║                                   ├── Demographic RAF coefficients   ║
║                                   ├── Age bracket binning            ║
║                                   └── Dual-eligible enrichment       ║
╚═══════════════════════════╦══════════════════════════════════════════╝
                            ║ AGGREGATION (member-level)
                            ▼
╔══════════════════════════════════════════════════════════════════════╗
║  GOLD LAYER  (analytics-ready, ML-ready, analyst-facing)            ║
║                                                                      ║
║  gold.member_raf_scores           gold.utilization_summary           ║
║  ├── Final RAF score              ├── IP admits / 1,000 members      ║
║  │   (demo + HCC + interactions)  ├── ED visits / 1,000 members      ║
║  ├── Risk tier: low/mod/high      ├── PMPM by member × period        ║
║  ├── Estimated annual cost        └── Specialist / PC visit rates    ║
║  ├── Pre / post PMPM                                                 ║
║  └── IP / ED utilization counts   gold.monthly_trends                ║
║                                   ├── Cohort × month × arm           ║
║  ml_features.risk_feature_store   ├── 3-month rolling avg PMPM       ║
║  ├── 33 engineered features       ├── DiD-ready structure            ║
║  ├── Log-transformed cost cols    └── IP / ED rates per 1,000        ║
║  ├── Pre-period utilization agg                                      ║
║  └── Boolean → int encoding                                          ║
║                                                                      ║
║  src/gold/shared_savings.py       ← MSSP calculator (testable API)   ║
║  ├── SharedSavingsResult          ├── gross_savings, savings_rate    ║
║  ├── SharedSavingsCalculator      ├── from_att(), project()          ║
║  └── compute_from_gold()          └── MSR gating, scenario table     ║
╚═══════════════════════════╦══════════════════════════════════════════╝
                            ║ TRAIN / EVALUATE / REGISTER
                            ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ML LAYER  (XGBoost + MLflow + SHAP + PSI drift monitoring)         ║
║                                                                      ║
║  src/ml/risk_model.py             ← Training, evaluation, registry   ║
║  ├── Classifier: risk_tier (low / moderate / high)                   ║
║  ├── XGBoost multiclass (objective: multi:softprob)                  ║
║  ├── Isotonic calibration via CalibratedClassifierCV (cv=3)          ║
║  ├── ECE: ~0.08 raw → < 0.03 post-calibration                       ║
║  ├── Regressor: estimated_annual_cost (XGBoost reg:squarederror)     ║
║  ├── TreeExplainer SHAP → beeswarm artifact logged to MLflow         ║
║  ├── Explicit ValueError on feature schema mismatch (no masking)     ║
║  └── Model Registry: auto-promote to Staging after training          ║
║                                                                      ║
║  src/ml/drift_monitor.py          ← Standalone PSI monitoring        ║
║  ├── compute_psi(): percentile binning, edge-case safe               ║
║  ├── FeatureDriftResult dataclass (feature, psi, status)             ║
║  ├── PSIReport: .summary(), .to_dict(), .to_dataframe()              ║
║  ├── .overall_status, .alerts, .critical, .max_psi properties        ║
║  ├── Auto-logs psi_{feature} metrics to active MLflow run            ║
║  └── PSI < 0.10 STABLE | 0.10–0.25 WARNING | > 0.25 CRITICAL 🚨     ║
╚═══════════════════════════╦══════════════════════════════════════════╝
                            ║
                            ▼
╔══════════════════════════════════════════════════════════════════════╗
║  GOVERNANCE  (Unity Catalog — role-based, lineage-tagged)           ║
║                                                                      ║
║  Catalog: medicare_lakehouse                                         ║
║  ├── Schema: bronze      → engineers (ALL PRIVILEGES)                ║
║  ├── Schema: silver      → engineers (ALL) + scientists (SELECT)     ║
║  ├── Schema: gold        → analysts (SELECT) + scientists (SELECT)   ║
║  └── Schema: ml_features → scientists (ALL) + analysts (SELECT)      ║
║                                                                      ║
║  TBLPROPERTIES: data_classification, pipeline_layer,                 ║
║                 derived_from, hcc_version, retention_days            ║
║                                                                      ║
║  Analyst views (no member-level identifiers):                        ║
║  ├── v_high_risk_summary    ├── v_cohort_trends                      ║
║  └── v_top_hcc_conditions                                            ║
╚═══════════════════════════╦══════════════════════════════════════════╝
                            ║
                            ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ORCHESTRATION  (Databricks Workflows — 4-task DAG)                 ║
║                                                                      ║
║  bronze_ingestion → silver_processing → gold_aggregation             ║
║                                              └─→ model_retrain       ║
║                                                                      ║
║  Schedule: daily 02:00 UTC  •  max_retries: 2 (Bronze), 1 (others) ║
║  Email alerting on failure  •  per-task timeout enforcement          ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 📦 Phase 1 — Data Engineering Foundation

### DAY 0 — Setup and Data Loading

Before running the pipeline, the synthetic CMS claims dataset is generated by `src/data_generator/cms_claims_generator.py`. This is not a trivial random number generator — it produces clinically realistic Medicare claims with:

- **27-condition ICD-10 catalogue** with prevalence-weighted sampling (morbid obesity at 8%, metastatic cancer at 2% — calibrated to real MA population epidemiology)
- **Age-adjusted condition loading** — older members receive more conditions via Poisson-distributed encounter counts
- **Realistic service type distribution**: IP admits (8%), ED (12%), specialist (25%), primary care (40%), lab (15%)
- **Intervention/control arm** assignment with a pre-specified cost reduction for DiD analysis
- **Deterministic output** — same seed always produces identical data, making pipeline tests reproducible

**Dataset produced:**

| Component | Volume | Notes |
|-----------|--------|-------|
| Beneficiaries | 10,000 | Ages 65–95, sex/dual calibrated to MA population |
| Claim lines | ~250,000 | 24 months history (Jan 2022 – Dec 2023) |
| ICD-10 codes | 27 conditions → 50 codes | Mapped to CMS HCC v28 |
| Service types | 5 categories | With realistic cost distributions per type |
| Intervention arms | 50/50 split | Control vs treated — parallel trends validated |

```python
# One-line generation and save to landing zone
from src.data_generator.cms_claims_generator import generate_and_save

members, claims = generate_and_save(
    output_dir           = "/tmp/medicare_lakehouse/raw",
    n_members            = 10_000,
    n_months             = 24,
    intervention_effect  = -420.0,   # PMPM cost reduction for treated arm
    seed                 = 42,
)
```

### DAY 1-2 — Bronze Ingestion

**Design principle: Bronze is append-only and immutable. It is the single source of truth for all downstream processing.**

`src/bronze/ingest_claims.py` implements four production concerns:

**1. Schema enforcement** — every incoming file is validated against a declared `StructType` before it touches any Delta table. Files that do not conform are rejected at ingestion with a logged error, not silently accepted and discovered as corrupt data three layers later.

**2. Idempotent ingestion** — before writing, the loader checks whether the current `_batch_id` already exists in the target table. Re-running the same pipeline job (after a failure, for example) does not create duplicate rows.

```python
def _batch_already_ingested(self, table_name, batch_id):
    if not table_exists(self.spark, table_name):
        return False
    return self.spark.table(table_name) \
        .filter(F.col("_batch_id") == batch_id) \
        .limit(1).count() > 0
```

**3. Audit columns** — every Bronze row carries `_ingested_at` (UTC timestamp), `_pipeline_layer` ('bronze'), and `_batch_id` (yyyyMMddHHmmss). These propagate through all downstream layers, providing end-to-end lineage from raw file byte to final risk score.

**4. Partition and ZORDER strategy** — claims are partitioned by `claim_year` and `claim_month`. Combined with `ZORDER BY bene_id`, a query for a specific member within a specific month reads a small fraction of the total data files rather than scanning the full table.

> 💡 **Why not MERGE for Bronze?** MERGE on Bronze would allow updates, which would destroy the audit trail. Every raw file that ever arrived must be queryable via Delta time travel. This is not a stylistic preference — it is a requirement for HIPAA audit trails and any regulatory investigation of payment accuracy.

---

## 🧪 Phase 2 — Clinical Enrichment (Silver)

### DAY 3-4 — Silver Processing: Clean, Deduplicate, Enrich

Silver is where raw claims become analytically meaningful. Three output tables are produced from two Bronze inputs.

#### `silver.clean_claims` — Quality-validated claims

**Deduplication logic**: When multiple rows share the same `(claim_id, bene_id, service_date)`, the row with the highest `paid_amount` is retained. This reflects the real-world pattern where claim re-submissions replace originals with corrected payment amounts.

```python
df.withColumn("_row_rank",
    F.row_number().over(
        Window.partitionBy("claim_id", "bene_id", "service_date")
              .orderBy(F.desc("paid_amount"))
    )
).filter(F.col("_row_rank") == 1)
```

**9-flag quality framework** — each row receives one boolean flag per rule and a composite `_quality_pass` column. Critically, failed rows are **retained** in `clean_claims` for audit review — they are simply excluded from HCC mapping and Gold aggregations:

| Flag | Rule | Reason for Exclusion |
|------|------|----------------------|
| `flag_null_claim_id` | claim_id IS NULL | Cannot be tracked or reconciled |
| `flag_null_bene_id` | bene_id IS NULL | Cannot be attributed to a member |
| `flag_negative_amount` | claim_amount < 0 | Likely credit adjustment — separate process |
| `flag_zero_amount` | claim_amount = 0 | Phantom claim — no actual service rendered |
| `flag_excessive_amount` | claim_amount > $500,000 | Likely data entry error |
| `flag_future_date` | service_date > today | System clock error |
| `flag_null_service_date` | service_date IS NULL | Untraceable chronologically |
| `flag_null_icd10` | icd10_primary IS NULL | Cannot be HCC-mapped |
| `flag_bronze_null` | bronze null flag set | Inherited from Bronze validation |

#### `silver.hcc_mapped_claims` — CMS HCC v28 enrichment

The HCC mapper (`src/silver/hcc_mapper.py`) implements the CMS risk adjustment logic used to set Medicare Advantage capitation rates. It is registered as a set of Spark UDFs for distributed execution across the cluster — each executor can perform ICD-10 → HCC lookups independently without driver bottleneck.

**Mapping logic:**
```
ICD-10-CM code → HCC category number → RAF coefficient
```

Each member's claims are aggregated to a **unique HCC list** (no double-counting within HCC category) and coefficients are summed. This replicates the CMS adjudication logic exactly.

**Interaction terms** add additional RAF weight when clinically significant co-morbidity combinations are present:

| Interaction | HCCs Involved | Additional RAF | Clinical Rationale |
|-------------|---------------|----------------|--------------------|
| CHF × Atrial Fibrillation | 85 + 96 | +0.175 | Combined cardiovascular burden substantially increases hospitalisation risk |
| CHF × Diabetes | 85 + {17,18,19} | +0.156 | Cardiac + metabolic complexity, competing treatment priorities |
| CKD × Diabetes | {134–138} + {17–19} | +0.156 | Renal + metabolic progression, accelerated disease course |

**Columns added per claim row:**

```
hcc_list               array<int>   All HCC numbers from this claim's ICD codes
hcc_raf_total          double       Sum of unique HCC RAF coefficients
primary_hcc            int          HCC of primary diagnosis
primary_hcc_desc       string       Human-readable condition label
hcc_burden_count       int          Number of distinct HCCs on this claim
has_chf                boolean      CHF (HCC 85)
has_afib               boolean      Atrial Fibrillation (HCC 96)
has_diabetes           boolean      Any diabetes HCC (17, 18, 19)
has_ckd                boolean      Any CKD HCC (134–138)
has_cancer             boolean      Any cancer HCC (8, 9, 11, 12)
has_copd               boolean      COPD (HCC 111)
has_metastatic         boolean      Metastatic cancer (HCC 8)
interaction_chf_afib   double       Interaction RAF contribution
interaction_chf_diabetes double
interaction_ckd_diabetes double
hcc_interaction_total  double       Sum of all interaction terms
```

> 🔑 **To use the official CMS mapping file**: replace `HCC_MAPPING` in `src/silver/hcc_mapper.py` with the ICD-10-CM to HCC crosswalk CSV from [CMS.gov](https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/risk-adjustors). The UDF structure and all downstream logic require no other changes.

---

## 🥇 Phase 3 — Analytics-Ready Gold Layer

### DAY 5-6 — Gold Aggregation: RAF Scores, Utilisation, Feature Store

Gold is the layer that business stakeholders and models consume directly. Every table is pre-aggregated, ZORDER-optimised, and governed by Unity Catalog access controls. No analyst query touches Silver.

#### `gold.member_raf_scores` — Primary analytics output

**Final RAF score formula:**

```
RAF = demographic_raf
    + Σ(unique HCC RAF coefficients across all claims)
    + Σ(interaction term RAF where co-morbidity combinations are present)
```

**Risk tier thresholds** (aligned with typical ACO care management programme criteria):

| Tier | RAF Threshold | Typical Care Programme Response |
|------|--------------|--------------------------------|
| `high` | RAF ≥ 2.0 | Intensive case management, dedicated care coordinator assigned |
| `moderate` | 1.2 ≤ RAF < 2.0 | Targeted disease management outreach programme |
| `low` | RAF < 1.2 | Standard care pathway, preventive services focus |

#### `gold.monthly_trends` — DiD-ready cohort analytics

This table is structured specifically for **Difference-in-Differences causal analysis** and requires no further reshaping to estimate the ATT. One row per `(service_year, service_month, intervention_arm)` with:

- PMPM cost and 3-month rolling average (computed via Spark Window function)
- IP admissions and ED visits per 1,000 members annualised
- Pre/post period flag relative to intervention start (January 2023)

The **parallel trends assumption** — that control and treatment groups would have evolved similarly absent the intervention — can be tested directly from this table by examining pre-period slope equality before fitting the DiD estimator.

#### `ml_features.risk_feature_store` — 33-feature ML matrix

The feature store is the direct, model-ready input to XGBoost training. Features are grouped into six engineering layers:

```
Feature Group           Count    Engineering Notes
──────────────────────  ─────    ─────────────────────────────────────────────
Demographic                 5    age, sex_male, dual_eligible, demographic_raf,
                                 age_bracket (5-year bins)
HCC burden                  3    max_hcc_burden, max_hcc_raf, max_interaction_raf
Clinical flags              8    has_chf, has_afib, has_diabetes, has_ckd,
                                 has_cancer, has_copd, has_depression, has_metastatic
Pre-period utilization    11    pre_pmpm, pre_ip_admits, pre_ed_visits,
                                 pre_specialist_visits, pre_primary_visits,
                                 pre_n_months, pre_total_cost, pre_avg_claim,
                                 pre_max_claim, pre_ip_rate, pre_ed_rate
RAF and cost                3    raf_score, estimated_annual_cost, actual_pmpm
Log-transformed             3    log_pre_pmpm, log_estimated_cost,
                                 log_pre_total_cost
──────────────────────  ─────
Total                      33    (+ risk_tier and intervention_arm as labels)
```

**Engineering decisions:**
- **Log transforms** on cost features: healthcare claim amounts are severely right-skewed. `log1p()` compresses the tail and makes XGBoost tree splits substantially more informative for the cost regression target
- **Pre-period features only** for utilization: using post-period utilization would introduce data leakage into the cost regression. The model is trained to predict risk prospectively — based only on what was observable before the intervention
- **Boolean → int encoding** at feature store build time: downstream model training receives a clean numeric matrix with no type casting overhead or silent encoding bugs

#### `src/gold/shared_savings.py` — MSSP calculator (testable standalone API)

The MSSP formula that the README's business context section leads with now has a dedicated, independently testable module — not buried in notebook comments or Gold config thresholds.

```python
from src.gold.shared_savings import SharedSavingsCalculator

calc = SharedSavingsCalculator()

# From pipeline actuals (reads gold.member_raf_scores directly)
result = calc.compute_from_gold(spark)
print(result)

# From DiD ATT estimate
result = calc.from_att(att_pmpm_annual=391.0, n_lives=50_000)
# → gross: $19,550,000 | savings_rate: 3.99% | earned: $9,775,000

# Scale projection table
df = calc.project(att_pmpm=391.0)
display(df)
```

`SharedSavingsResult` auto-computes `gross_savings`, `savings_rate`, `earned_savings`, `per_member_gross`, `exceeds_msr`, and `status` in `__post_init__`. The widget-driven scenario explorer in notebook 03 calls `from_att()` live as sliders change — letting you model any combination of ATT, sharing rate, MSR, and attributed lives interactively.

| Method | Input | Use case |
|--------|-------|----------|
| `compute()` | benchmark_pmpm, actual_pmpm, n_lives | Pipeline actuals |
| `from_att()` | att_pmpm_annual, n_lives | DiD causal estimate |
| `project()` | att_pmpm | Scale comparison table |
| `compute_from_gold()` | spark session | Live Gold query |

---

## 🤖 Phase 4 — ML Risk Stratification

### DAY 7 — MLflow: Train, Track, Register, Explain, Monitor

#### Model architecture

Two XGBoost models are trained simultaneously against the same 33-feature matrix:

**Classifier (risk tier — three-class):**
```python
XGBClassifier(objective="multi:softprob", num_class=3)
    → CalibratedClassifierCV(method="isotonic", cv=3)
```

**Regressor (annual cost prediction):**
```python
XGBRegressor(objective="reg:squarederror")
```

#### Why isotonic calibration matters in healthcare

Raw XGBoost `predict_proba()` outputs are **not** reliable probability estimates — the model is systematically overconfident. In a clinical setting, a score of "0.85 probability high-risk" is used to trigger a care management workflow. If that 0.85 is empirically a 0.62, care managers are being deployed to members who do not need intensive intervention — wasting resources and potentially harming programme economics.

Isotonic regression post-processes the raw outputs without assuming a functional form:

```
Before calibration:  Expected Calibration Error (ECE) ≈ 0.08
After calibration:   Expected Calibration Error (ECE) < 0.03
```

This is not optional in a regulated healthcare setting. It is the difference between a model that is clinically deployable and one that is not.

#### SHAP explainability — the clinical trust layer

Every member flagged as high-risk receives a SHAP waterfall showing exactly which features drove that classification and by how much. This is not cosmetic — it is what allowed the Uganda Ministry of Health to trust and scale the companion immunization defaulter model from 2 to 5 districts after validation. Clinicians can interrogate the reason for every prediction before acting on it.

```python
# Extract SHAP values from base XGBoost booster (inside CalibratedClassifierCV)
base_model = model.clf.calibrated_classifiers_[0].estimator
explainer  = shap.TreeExplainer(base_model)
shap_vals  = explainer.shap_values(X_sample)
# Result: per-member, per-feature contribution to high-risk probability
```

**Artifacts logged to MLflow per training run:**

| Artifact | Content |
|----------|---------|
| `shap_importance.png` | Beeswarm of top 20 features by mean absolute SHAP value |
| `classification_report.json` | Precision / recall / F1 per risk tier |
| `feature_list.txt` | Exact feature columns (versioned with the model) |
| `xgboost_classifier` | Native XGBoost booster for inference |
| `calibrated_classifier` | Full sklearn calibrated pipeline |
| `cost_regressor` | XGBoost cost regression model |

#### MLflow Model Registry lifecycle

```
Training run
    └── model logged to experiment with full params + metrics
           └── registered to Model Registry as version N
                  └── automatically promoted to "Staging"
                         └── (manual approval) promoted to "Production"
                                └── previous Production version archived
```

#### PSI drift monitoring — `src/ml/drift_monitor.py`

Extracted into its own module so the PSI logic is independently testable without standing up a model training run. Returns a structured `PSIReport` rather than a raw dict:

```python
from src.ml.drift_monitor import monitor_drift, DEFAULT_DRIFT_FEATURES

report = monitor_drift(
    baseline_table      = "medicare_lakehouse.ml_features.risk_feature_store",
    current_table       = "medicare_lakehouse.ml_features.risk_feature_store_latest",
    features_to_monitor = DEFAULT_DRIFT_FEATURES,
    spark               = spark,
)

print(report.summary())          # human-readable status table
display(report.to_dataframe())   # sortable DataFrame for notebook
mlflow.log_metrics(report.to_dict())  # psi_{feature} scalars per run
```

`PSIReport` properties: `.overall_status` ("STABLE" / "WARNING" / "CRITICAL"), `.alerts` (list of drifting features), `.critical`, `.n_alerts`, `.max_psi`. Auto-logs `psi_{feature}` metrics to the active MLflow run when called during training.

```
PSI < 0.10    STABLE   — population consistent with baseline
PSI 0.10–0.25 WARNING  — moderate shift, investigate root cause
PSI > 0.25    CRITICAL — significant drift, retrain immediately
```

**Features monitored by default:** `raf_score`, `pre_pmpm`, `max_hcc_burden`, `max_hcc_raf`, `pre_ip_admits`, `pre_ed_visits`, `estimated_annual_cost`, `demographic_raf`

---

## 🔒 Phase 5 — Governance and Orchestration

### DAY 8 — Unity Catalog and Databricks Workflows

#### Unity Catalog access model

The access control design mirrors a real ACO data environment with four role groups:

```
Role              Bronze    Silver    Gold      ML Features
────────────────  ───────   ───────   ───────   ───────────
analysts          ✗         ✗         SELECT    SELECT
engineers         ALL       ALL       SELECT    ✗
data scientists   ✗         SELECT    SELECT    ALL
auditors          SELECT    SELECT    SELECT    SELECT
```

**Why analysts cannot access Silver:** Silver contains claim-line detail with `bene_id` and ICD-10 codes. The access model reflects HIPAA minimum-necessary principles. Analysts consume pre-aggregated Gold views — they have no business need for individual claim lines, and access should not be granted without a documented use case.

**Table lineage tags via TBLPROPERTIES:**
```sql
ALTER TABLE gold.member_raf_scores SET TBLPROPERTIES (
    'data_classification' = 'HIPAA_deidentified',
    'pipeline_layer'      = 'gold',
    'hcc_version'         = 'v28',
    'derived_from'        = 'medicare_lakehouse.silver.hcc_mapped_claims',
    'retention_days'      = '2555'   -- 7 years per HIPAA
);
```

**Three analyst-facing Gold views** (no member-level identifiers):

| View | Content |
|------|---------|
| `v_high_risk_summary` | Tier rollup: count, mean RAF, mean PMPM, IP/ED totals |
| `v_cohort_trends` | Monthly PMPM, rolling average, IP/ED rates by intervention arm |
| `v_top_hcc_conditions` | Top 20 conditions by prevalence, member count, mean RAF |

#### Databricks Workflow — 4-task DAG

```
[bronze_ingestion]
      │  timeout: 30 min  │  max_retries: 2  │  retry_on_timeout: true
      ▼
[silver_processing]
      │  timeout: 40 min  │  max_retries: 1
      ▼
[gold_aggregation]
      │  timeout: 40 min  │  max_retries: 1
      ▼
[model_retrain]
         timeout: 60 min  │  max_retries: 1
         email on failure: keyegon@gmail.com
```

**Schedule:** Daily at 02:00 UTC (off-peak for cluster cost efficiency)

**To import:** `Workflows → Create Job → Edit JSON` → paste `workflows/pipeline_workflow.json` → replace `REPLACE_WITH_YOUR_CLUSTER_ID`

---

## 🧠 Key Technical Decisions

### 1. Append-only Bronze — not MERGE

MERGE on Bronze implies updates, which destroys the audit trail. Every raw file must be queryable via Delta time travel at any future point — required for HIPAA audit trails and regulatory investigation of payment accuracy. Bronze is write-once, read-always.

### 2. Isotonic calibration over Platt scaling

Platt scaling (logistic regression on model outputs) assumes a sigmoid relationship between raw scores and true probabilities. Healthcare risk scores rarely follow a sigmoid curve — the relationship is nonlinear and dataset-specific. Isotonic regression makes no shape assumption and fits a monotone step function to the empirical calibration data. It is strictly more accurate for this use case; the cost is needing more calibration samples (addressed by 3-fold cross-validation).

### 3. PSI over KS test for drift monitoring

The KS test detects whether two distributions differ but does not quantify the magnitude of the shift. PSI produces a scalar that maps directly to interpretable action levels, is additive across features, and is the established industry standard in credit risk and healthcare actuarial analytics. A PSI of 0.18 means "moderate drift, investigate." A KS p-value of 0.03 means nothing to a care management director.

### 4. Pre-period features only in the feature store

Using post-period utilization as predictors when the model is trained to predict cost would introduce direct data leakage. The feature store is built to support **prospective** risk identification — predicting who will become high-cost before the event occurs. Post-period data belongs in the outcome measurement, not the feature matrix.

### 5. Medallion over a single wide table

| Concern | Single Wide Table | Medallion Architecture |
|---------|------------------|----------------------|
| Silver failure propagates downstream | Yes | No — Bronze is untouched |
| Can re-run Gold without re-ingesting | No | Yes |
| Analyst query performance | Full scan of raw detail | Pre-aggregated Gold |
| PHI-adjacent data exposure | All users see everything | RBAC per schema |
| Time travel per processing stage | One shared history | Independent version per layer |
| Data lineage | Opaque | Tagged via TBLPROPERTIES |
| Recovery from bad Silver logic | Re-process everything | Re-run Silver only |

### 6. ZORDER by bene_id on all claim tables

Healthcare analytics queries are almost always member-centric — "all claims for this member," "this member's risk score," "this member's utilization." `ZORDER BY bene_id` co-locates all claim rows for the same member in the same Parquet files. Combined with year/month partitioning, a per-member query reads a fraction of the total files rather than a full table scan.

---

## 📊 Performance Benchmarks

### Risk Tier Classification (XGBoost + Isotonic Calibration)

Results on 10,000-member synthetic cohort — 80/20 stratified train/test split:

| Metric | Value | Notes |
|--------|-------|-------|
| Tier accuracy | ~88–92% | Stratified on risk_tier |
| AUC (macro OvR) | ~0.96–0.98 | One-vs-rest across all three tiers |
| Expected Calibration Error | < 0.03 | After isotonic calibration |
| High-risk precision | ~91% | Minimising false positives |
| High-risk recall | ~89% | Minimising missed high-risk members |

### Annual Cost Regression (XGBoost)

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | ~$400–700 / member / year |
| R² | ~0.85–0.92 |
| RMSE | ~$900–1,400 |

### Pipeline Runtime (Databricks Community Edition)

| Stage | Runtime |
|-------|---------|
| Bronze ingestion (10K members, 24 months) | ~3 min |
| Silver processing (dedup + HCC mapping) | ~5 min |
| Gold aggregation (RAF + feature store) | ~5 min |
| MLflow model training + SHAP | ~5 min |
| Governance setup | ~2 min |
| **Total end-to-end** | **~20 min** |

> ⚠️ **Synthetic data note:** Tier accuracy is higher here than on real Medicare claims because the generator assigns condition loadings that make tiers separable by construction. Real-world moderate/high boundary confusion would be greater. This is documented in the model card produced per MLflow run.

---

## 📖 Data Dictionary

### `gold.member_raf_scores` — Primary analytics output

| Column | Type | Description |
|--------|------|-------------|
| `bene_id` | string | Medicare beneficiary identifier |
| `age` | int | Member age at analysis date |
| `sex` | string | Biological sex: F / M |
| `dual_eligible` | int | 1 = Medicare + Medicaid dual eligible |
| `plan_type` | string | HMO / PPO / PFFS / SNP |
| `state` | string | State of residence |
| `demographic_raf` | double | CMS v28 age-sex RAF coefficient |
| `max_hcc_burden` | int | Peak number of distinct HCCs on any single claim |
| `max_hcc_raf` | double | Maximum HCC RAF total across all claims |
| `max_interaction_raf` | double | Maximum interaction term RAF |
| `raf_score` | **double** | **Final RAF: demographic + HCC + interactions** |
| `risk_tier` | string | low / moderate / high |
| `estimated_annual_cost` | double | RAF × $9,800 benchmark × 12 months |
| `actual_pmpm` | double | Actual per-member-per-month from claims |
| `pre_pmpm` | double | PMPM in pre-intervention period (2022) |
| `post_pmpm` | double | PMPM in post-intervention period (2023) |
| `ip_admit_count` | long | Total inpatient admissions over study period |
| `ed_visit_count` | long | Total ED visits over study period |
| `total_claims` | long | Total claim lines over study period |
| `intervention_arm` | int | 0 = control group, 1 = intervention group |
| `has_chf` | boolean | Congestive Heart Failure present (HCC 85) |
| `has_afib` | boolean | Atrial Fibrillation present (HCC 96) |
| `has_diabetes` | boolean | Any diabetes HCC (17, 18, or 19) |
| `has_ckd` | boolean | Any CKD HCC (134–138) |
| `has_cancer` | boolean | Any cancer HCC (8, 9, 11, or 12) |
| `_ingested_at` | timestamp | Pipeline write timestamp (UTC) |
| `_pipeline_layer` | string | 'gold' |
| `_batch_id` | string | yyyyMMddHHmmss run identifier |

### HCC v28 Condition Catalogue (50 ICD-10 codes mapped)

| Condition Group | HCC Range | Representative ICD-10 | RAF Range |
|-----------------|-----------|----------------------|-----------|
| Cardiovascular | 85, 86, 96, 107, 108 | I500, I480, I2109 | 0.178 – 0.421 |
| Diabetes | 17, 18, 19 | E1140, E119, E1010 | 0.118 – 0.302 |
| Renal Disease | 134–138 | N183–N185, N19, Z992 | 0.071 – 0.289 |
| Pulmonary | 110, 111 | J449, J84189 | 0.245 – 0.335 |
| Cancer | 8, 9, 11, 12 | C7951, C349, C189 | 0.150 – 2.488 |
| Neurological / MH | 40, 57, 58, 77–79 | G20, G35, F329, F209 | 0.406 – 0.625 |
| Other | 22 | E6601 | 0.178 |

---

## 📁 Repository Structure

32 files — one responsibility per module, one class per file.

```
databricks-medicare-lakehouse/
│
├── 📄 README.md                             ← You are here
├── 📄 requirements.txt                      ← pip dependencies
│
├── ⚙️  config/
│   └── pipeline_config.py                  ← Single source of truth for every path,
│                                               table name, ML hyperparameter, and
│                                               business threshold in the pipeline.
│                                               No magic strings anywhere else.
│
├── 🧠 src/                                  ← One class per file, one concern per class.
│   │
│   ├── data_generator/
│   │   └── cms_claims_generator.py         ← Synthetic CMS claims + member demographics.
│   │                                           27-condition ICD-10 catalogue, prevalence-
│   │                                           weighted sampling, age-adjusted condition
│   │                                           loading, intervention/control arm assignment.
│   │                                           Deterministic via seed — same seed = same data.
│   ├── utils/
│   │   └── spark_utils.py                  ← SparkSession factory, Delta write/upsert/
│   │                                           time-travel helpers, StructType schema
│   │                                           validation, audit column injection,
│   │                                           null counting, quality flag utilities.
│   ├── bronze/
│   │   └── ingest_claims.py                ← Append-only ingestion with _batch_id
│   │                                           idempotency guard, StructType schema
│   │                                           enforcement, Bronze null flagging,
│   │                                           ZORDER + OPTIMIZE post-write.
│   ├── silver/
│   │   ├── hcc_mapper.py                   ← ICD-10-CM → CMS HCC v28 Spark UDFs
│   │   │                                      (50 codes, distributed execution).
│   │   │                                      RAF coefficient aggregation with no
│   │   │                                      double-counting. CHF×AFib (+0.175),
│   │   │                                      CKD×Diabetes (+0.156) interaction terms.
│   │   │                                      8 boolean condition flags per claim.
│   │   └── clean_claims.py                 ← Dedup (row_number, max paid_amount wins).
│   │                                           9 quality flags + _quality_pass column.
│   │                                           Date/service feature extraction.
│   │                                           Member profile join.
│   ├── gold/
│   │   ├── raf_aggregates.py               ← Member RAF scores (demo + HCC + interactions).
│   │   │                                      Risk tier assignment (low/moderate/high).
│   │   │                                      Utilization summary (IP, ED, PMPM by period).
│   │   │                                      Monthly DiD-ready cohort trends with 3-month
│   │   │                                      rolling average. 33-feature ML feature store
│   │   │                                      with pre-period-only utilization features,
│   │   │                                      log transforms, boolean→int encoding.
│   │   └── shared_savings.py               ← MSSP Shared Savings Programme calculator.
│   │                                           SharedSavingsResult dataclass (auto-computes
│   │                                           gross savings, savings rate, earned savings,
│   │                                           MSR gating). SharedSavingsCalculator with
│   │                                           compute(), from_att(), project(), and
│   │                                           compute_from_gold() methods. Independently
│   │                                           unit-testable — no Spark or MLflow required.
│   ├── ml/
│   │   ├── risk_model.py                   ← RiskStratificationModel: XGBoost multiclass
│   │   │                                      classifier + isotonic calibration (ECE < 0.03)
│   │   │                                      + XGBoost cost regressor. Full MLflow run:
│   │   │                                      params, metrics, SHAP beeswarm artifact,
│   │   │                                      classification report, feature list. Model
│   │   │                                      Registry promotion to Staging. Explicit
│   │   │                                      ValueError on feature schema mismatch —
│   │   │                                      no silent fallback masking misconfiguration.
│   │   └── drift_monitor.py                ← PSI drift monitoring extracted as standalone
│   │                                           module. compute_psi() with percentile binning,
│   │                                           edge-case safety (empty series, degenerate
│   │                                           distributions). FeatureDriftResult dataclass.
│   │                                           PSIReport with .summary(), .to_dict(),
│   │                                           .to_dataframe(), .overall_status, .alerts,
│   │                                           .critical properties. Auto-logs to active
│   │                                           MLflow run. DEFAULT_DRIFT_FEATURES list.
│   │                                           No model training dependency — run standalone.
│   └── governance/
│       └── unity_catalog_setup.py          ← Catalog + schema creation. GRANT/REVOKE
│                                               by role group (analysts/engineers/scientists/
│                                               auditors). TBLPROPERTIES lineage tagging
│                                               (data_classification, derived_from,
│                                               hcc_version, retention_days). Three
│                                               analyst-facing Gold views with no member IDs.
│
├── 📓 notebooks/                            ← Databricks notebooks — import via Repos.
│   │                                           Self-contained, widget-driven, SQL validation
│   │                                           cells throughout. Run in numbered order.
│   ├── 01_bronze_ingestion.py              ← Widget params (n_members, n_months, batch_id).
│   │                                           Generate data → save to landing zone →
│   │                                           ingest to Bronze Delta → verify schema,
│   │                                           partition stats, null summary.
│   ├── 02_silver_processing.py             ← Quality flag breakdown (per-flag counts).
│   │                                           HCC condition co-occurrence analysis.
│   │                                           Member profile demographics. Delta time
│   │                                           travel demo (read version 0).
│   ├── 03_gold_aggregates.py               ← RAF distribution by tier. Manual DiD estimate
│   │                                           from monthly_trends. Feature correlation
│   │                                           heatmap. MSSP shared savings projection
│   │                                           via SharedSavingsCalculator — widget-driven
│   │                                           scenario explorer (ATT, sharing rate, n_lives).
│   ├── 04_mlflow_model.py                  ← Feature store load. Train + evaluate.
│   │                                           Confusion matrix + predicted vs actual cost.
│   │                                           MLflow experiment run history table. PSI
│   │                                           drift report via PSIReport.summary() +
│   │                                           .to_dataframe(). Model Registry version list.
│   └── 05_governance_and_workflows.py      ← Unity Catalog setup. Analyst view validation.
│                                               TBLPROPERTIES lineage display. Delta OPTIMIZE
│                                               + VACUUM maintenance. Workflow JSON (4-task
│                                               DAG ready to paste into Databricks Jobs UI).
│                                               Full end-to-end lineage diagram in comments.
│
├── ⚡ workflows/
│   └── pipeline_workflow.json              ← Databricks Workflow definition. 4-task DAG:
│                                               bronze_ingestion → silver_processing →
│                                               gold_aggregation → model_retrain. Daily
│                                               02:00 UTC. Per-task timeouts + retries.
│                                               Email alert on failure. Replace
│                                               REPLACE_WITH_YOUR_CLUSTER_ID to activate.
│
├── 🧪 tests/                               ← Pure-Python tests. No Spark or MLflow needed.
│   ├── test_generator.py                   ← 20+ assertions: member count, age range,
│   │                                           unique IDs, determinism, claim validity,
│   │                                           service types, HCC mapping accuracy,
│   │                                           RAF coefficient totals, no double-counting,
│   │                                           interaction term directionality.
│   ├── test_drift_and_savings.py           ← 35 assertions across SharedSavingsResult,
│   │                                           SharedSavingsCalculator, compute_psi,
│   │                                           _classify_psi, FeatureDriftResult, PSIReport.
│   │                                           Edge cases: empty series, degenerate
│   │                                           distributions, MSR gating, sharing rate
│   │                                           overrides, proportional scaling, 50K-member
│   │                                           pipeline validation vs README numbers.
│   └── __init__.py
│
└── 📚 docs/
    └── SETUP.md                            ← Step-by-step Databricks setup guide.
                                               Cluster requirements, dependency install,
                                               notebook execution order, Workflow import,
                                               Community Edition compatibility notes.
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Databricks workspace | Community Edition | Standard or Premium |
| Databricks Runtime | 13.3 LTS | 14.x LTS |
| Python | 3.9 | 3.11 |
| Cluster node type | i3.xlarge | i3.2xlarge |
| Unity Catalog | Optional | Required for full GRANT governance |

### Step 1 — Clone into Databricks Repos

```
Databricks UI → Workspace → Repos → Add Repo
URL: https://github.com/erickyegon/databricks-medicare-lakehouse
```

### Step 2 — Install dependencies

```python
# Run in any notebook cell, or add to cluster init script
%pip install xgboost shap scikit-learn mlflow delta-spark
dbutils.library.restartPython()
```

### Step 3 — Run notebooks in order

| # | Notebook | Purpose | Runtime |
|---|----------|---------|---------|
| 01 | `01_bronze_ingestion` | Generate synthetic data → Bronze Delta | ~3 min |
| 02 | `02_silver_processing` | Clean, dedup, HCC-map → Silver Delta | ~5 min |
| 03 | `03_gold_aggregates` | RAF scores, feature store → Gold Delta | ~5 min |
| 04 | `04_mlflow_model` | Train XGBoost, log to MLflow, register | ~5 min |
| 05 | `05_governance_and_workflows` | Unity Catalog + Workflow definition | ~2 min |

### Step 4 — Import the Workflow

```
Workflows → Create Job → Edit JSON
→ paste contents of workflows/pipeline_workflow.json
→ replace REPLACE_WITH_YOUR_CLUSTER_ID
→ Save → Run Now
```

### Step 5 — Explore the results

```sql
-- RAF distribution by risk tier
SELECT risk_tier,
       COUNT(*)                        AS n_members,
       ROUND(AVG(raf_score), 3)        AS mean_raf,
       ROUND(AVG(actual_pmpm), 2)      AS mean_pmpm,
       SUM(ip_admit_count)             AS total_ip_admits
FROM medicare_lakehouse.gold.member_raf_scores
GROUP BY risk_tier ORDER BY mean_raf DESC;

-- DiD check: pre vs post PMPM by intervention arm
SELECT period, intervention_arm,
       ROUND(AVG(pmpm_cost), 2)        AS mean_pmpm,
       COUNT(DISTINCT service_month)   AS n_months
FROM medicare_lakehouse.gold.monthly_trends
GROUP BY period, intervention_arm
ORDER BY period DESC, intervention_arm;

-- Top conditions by prevalence (analyst view — no member IDs)
SELECT * FROM medicare_lakehouse.gold.v_top_hcc_conditions;
```

---

## 🔧 Troubleshooting

**`ModuleNotFoundError: No module named 'delta'`**
```python
%pip install delta-spark
dbutils.library.restartPython()
```

**`GRANT` statements failing on Community Edition**  
Unity Catalog GRANT/REVOKE requires a UC-enabled workspace. On Community Edition, these statements are caught and logged as warnings — all Delta tables, MLflow tracking, and pipeline logic work normally.

**`Table already exists` on re-run**  
All Gold and Silver writes use `mode="overwrite"`. Bronze uses `mode="append"` with `_batch_id` idempotency. Re-running any notebook is safe by design.

**Bronze ingestion reports 0 rows written**  
The `_batch_id` idempotency guard is active — this batch was already ingested. Leave the `batch_id` widget blank to auto-generate a new timestamp-based ID.

**MLflow experiment not found**  
Ensure `/Shared/medicare_raf_risk_stratification` exists or update `experiment_name` in `config/pipeline_config.py` to a path you have write access to.

---

## 🔗 Related Portfolio Projects

This project is the **data engineering foundation** of a broader healthcare AI portfolio. The Lakehouse infrastructure built here feeds the analytics and models in the companion repositories:

| Project | Description | Key Output |
|---------|-------------|-----------|
| [medicare-raf-prototypes](https://github.com/erickyegon/medicare-raf-prototypes) | XGBoost + SHAP risk adjustment + Streamlit dashboard + DiD causal attribution | ATT = −$391/member (p < 0.0001) |
| [clinical-doc-intelligence](https://github.com/erickyegon/clinical-doc-intelligence) | LangChain + LangGraph RAG for FDA drug labels with PHI detection and RAGAS evaluation | 54 automated tests passing |
| [prior-auth-dss](https://github.com/erickyegon/prior-auth-dss) | LangGraph multi-agent prior authorization decision support with RLHF alignment | End-to-end autonomous clinical review |
| [immunization-defaulter-risk-engine](https://github.com/erickyegon/immunization-defaulter-risk-engine) | XGBoost + SHAP defaulter prediction scaled to 5 Uganda MOH districts | ECE = 0.023 post-calibration |

---

## 👤 Author

**Erick Kiprotich Yegon, PhD**  
*Lead Data Scientist · Healthcare AI · LLMOps · Agentic Systems*

PhD in Epidemiology (WES-verified U.S. equivalent) · 17+ years in production ML and healthcare analytics · 30+ peer-reviewed publications including The Lancet Global Health (h-index 10) · U.S. Permanent Resident (EB-1A Extraordinary Ability)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-erickyegon-0A66C2?logo=linkedin)](https://linkedin.com/in/erickyegon)
[![GitHub](https://img.shields.io/badge/GitHub-erickyegon-181717?logo=github)](https://github.com/erickyegon)
[![YouTube](https://img.shields.io/badge/YouTube-DataStride-FF0000?logo=youtube)](https://youtube.com/@DataStride)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--7055--4848-A6CE39?logo=orcid)](https://orcid.org/0000-0002-7055-4848)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*This project uses entirely synthetic data generated for pipeline demonstration purposes. No real Medicare beneficiary data is used or represented. ICD-10 codes and HCC mappings reflect the CMS v28 regulatory framework, included for technical accuracy only.*
