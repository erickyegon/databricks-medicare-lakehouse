"""
src/data_generator/cms_claims_generator.py
──────────────────────────────────────────
Generates realistic synthetic CMS-style Medicare claims and member
demographics for pipeline development and testing.

Design principles:
  - Reproduces real CMS data structure (claim header + line items)
  - ICD-10-CM codes drawn from a weighted catalogue of common HCC conditions
  - Demographic distributions calibrated to real Medicare Advantage population
  - Intervention/control arm assignment for DiD-style causal analysis
  - Deterministic via seed — same seed = same data every run

Output tables:
  - claims:  one row per claim-line (header joined)
  - members: one row per beneficiary
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Tuple

from src.utils.spark_utils import get_logger

logger = get_logger(__name__)


# ── ICD-10 / HCC Catalogue ─────────────────────────────────────────────────
# Format: (icd10_code, hcc_number, condition_label, raf_coefficient, prevalence_weight)

ICD10_CATALOGUE = [
    # Cardiovascular
    ("I500",  85,  "Congestive Heart Failure",            0.323, 0.08),
    ("I480",  96,  "Atrial Fibrillation",                 0.421, 0.10),
    ("I2109", 86,  "Acute Myocardial Infarction",         0.218, 0.05),
    ("I7001", 107, "Vascular Disease w/ Complications",   0.299, 0.04),
    ("I739",  108, "Vascular Disease",                    0.178, 0.07),
    # Diabetes
    ("E1140", 18,  "T2DM with CKD/Angiopathy",            0.302, 0.09),
    ("E119",  19,  "T2DM Without Complications",           0.118, 0.15),
    ("E1010", 17,  "T1DM Acute Complication",              0.302, 0.02),
    # Renal
    ("N183",  138, "CKD Stage 3",                          0.071, 0.08),
    ("N184",  137, "CKD Stage 4",                          0.138, 0.04),
    ("N185",  136, "CKD Stage 5",                          0.143, 0.02),
    ("N19",   135, "Renal Failure",                        0.289, 0.03),
    ("Z992",  134, "Dialysis Status",                      0.289, 0.02),
    # Pulmonary
    ("J449",  111, "COPD",                                 0.245, 0.10),
    ("J84189",110, "Cystic Fibrosis / ILD",               0.335, 0.01),
    # Cancer
    ("C7951", 8,   "Metastatic Cancer",                    2.488, 0.02),
    ("C349",  9,   "Lung Cancer",                          0.899, 0.02),
    ("C189",  11,  "Colorectal Cancer",                    0.439, 0.03),
    ("C509",  12,  "Breast Cancer",                        0.150, 0.04),
    # Neurological
    ("G409",  79,  "Seizure Disorders",                    0.448, 0.03),
    ("G20",   78,  "Parkinson's Disease",                  0.406, 0.02),
    ("G35",   77,  "Multiple Sclerosis",                   0.597, 0.01),
    ("M0500", 40,  "Rheumatoid Arthritis",                 0.455, 0.04),
    ("F329",  58,  "Major Depression",                     0.421, 0.06),
    ("F209",  57,  "Schizophrenia",                        0.625, 0.02),
    # Other
    ("E6601", 22,  "Morbid Obesity",                       0.178, 0.08),
    # Healthy / low-risk (visit with no HCC)
    ("Z0000", None, "Routine encounter",                   0.000, 0.30),
]

ICD10_DF = pd.DataFrame(ICD10_CATALOGUE, columns=[
    "icd10_code", "hcc_number", "condition_label", "raf_coefficient", "prevalence_weight"
])

# Normalize weights
ICD10_DF["_w"] = ICD10_DF["prevalence_weight"] / ICD10_DF["prevalence_weight"].sum()

# Service type catalogue
SERVICE_TYPES = {
    "ip_admit":   {"weight": 0.08, "base_cost": (8_000, 25_000)},
    "ed_visit":   {"weight": 0.12, "base_cost": (800,   3_500)},
    "specialist": {"weight": 0.25, "base_cost": (150,   600)},
    "primary":    {"weight": 0.40, "base_cost": (80,    250)},
    "lab":        {"weight": 0.15, "base_cost": (30,    400)},
}

# Payer types
PLAN_TYPES = ["HMO", "PPO", "PFFS", "SNP"]

# Provider specialty codes (simplified)
SPECIALTY_CODES = ["01", "06", "11", "38", "78", "93", "99"]


# ── Demographic generators ─────────────────────────────────────────────────

def _generate_members(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic Medicare beneficiary demographics."""
    bene_ids = [f"BNE{str(i).zfill(8)}" for i in range(1, n + 1)]

    ages   = rng.integers(65, 96, size=n)
    sexes  = rng.choice(["F", "M"], size=n, p=[0.56, 0.44])
    duals  = rng.choice([0, 1], size=n, p=[0.80, 0.20])
    plans  = rng.choice(PLAN_TYPES, size=n)

    # Enrollment start: 1–5 years before baseline
    days_enrolled = rng.integers(365, 1826, size=n)
    baseline = date(2022, 1, 1)
    enroll_dates = [
        (baseline - timedelta(days=int(d))).isoformat()
        for d in days_enrolled
    ]

    # Intervention arm (50/50 random assignment)
    intervention = rng.choice([0, 1], size=n, p=[0.50, 0.50])

    # State / county (simplified)
    states = rng.choice(
        ["KY", "OH", "TN", "IN", "WV", "VA", "NC", "GA"],
        size=n, p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.10, 0.05]
    )

    return pd.DataFrame({
        "bene_id":          bene_ids,
        "age":              ages,
        "sex":              sexes,
        "dual_eligible":    duals,
        "plan_type":        plans,
        "state":            states,
        "enrollment_date":  enroll_dates,
        "intervention_arm": intervention,
    })


# ── Condition assignment ───────────────────────────────────────────────────

def _assign_conditions(
    member: pd.Series,
    rng: np.random.Generator,
    n_conditions_range: Tuple[int, int] = (1, 6),
) -> list:
    """
    Sample ICD-10 codes for a member, weighted by prevalence.
    Older / sicker members get more conditions.
    """
    age_factor = max(1, (member["age"] - 65) // 10)
    n_conditions = int(rng.integers(
        n_conditions_range[0],
        min(n_conditions_range[1] + age_factor, len(ICD10_DF)),
    ))
    chosen = ICD10_DF.sample(n=n_conditions, weights="_w", replace=True, random_state=rng.integers(1e6)).drop_duplicates(subset=["icd10_code"])
    return chosen["icd10_code"].tolist()


# ── Claim generator ────────────────────────────────────────────────────────

def _generate_claims_for_member(
    member: pd.Series,
    service_months: list,
    intervention_effect: float,
    rng: np.random.Generator,
) -> list:
    """Generate a list of claim-line dicts for one member across all months."""
    rows = []
    icd_codes = _assign_conditions(member, rng)

    for month_start in service_months:
        # Number of encounters this month (higher for sicker members)
        n_encounters = max(1, int(rng.poisson(lam=2.5 + 0.5 * len(icd_codes))))

        for enc_idx in range(n_encounters):
            # Service type
            svc_names   = list(SERVICE_TYPES.keys())
            svc_weights = [v["weight"] for v in SERVICE_TYPES.values()]
            svc_type    = rng.choice(svc_names, p=np.array(svc_weights) / sum(svc_weights))
            svc_info    = SERVICE_TYPES[svc_type]

            # Service date within the month
            days_in_month = 28
            service_date  = month_start + timedelta(days=int(rng.integers(0, days_in_month)))

            # Cost — lower for intervention arm after intervention start
            base_low, base_high = svc_info["base_cost"]
            cost = rng.uniform(base_low, base_high)
            if member["intervention_arm"] == 1 and month_start >= date(2023, 1, 1):
                cost = max(0, cost + intervention_effect / 12 / n_encounters)

            # Claim ID
            claim_id = (
                f"CLM{member['bene_id']}"
                f"{month_start.strftime('%Y%m')}"
                f"{str(enc_idx).zfill(3)}"
            )

            # ICD-10 for this encounter
            enc_icd = rng.choice(icd_codes)

            rows.append({
                "claim_id":          claim_id,
                "bene_id":           member["bene_id"],
                "service_date":      service_date.isoformat(),
                "claim_year":        service_date.year,
                "claim_month":       service_date.month,
                "service_type":      svc_type,
                "icd10_primary":     enc_icd,
                "icd10_codes":       "|".join(icd_codes),
                "provider_specialty":rng.choice(SPECIALTY_CODES),
                "claim_amount":      round(float(cost), 2),
                "allowed_amount":    round(float(cost) * rng.uniform(0.75, 0.95), 2),
                "paid_amount":       round(float(cost) * rng.uniform(0.60, 0.90), 2),
                "plan_type":         member["plan_type"],
                "intervention_arm":  int(member["intervention_arm"]),
            })
    return rows


# ── Public API ─────────────────────────────────────────────────────────────

def generate_members(
    n: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic Medicare beneficiary cohort."""
    rng = np.random.default_rng(seed)
    members = _generate_members(n, rng)
    logger.info(f"Generated {len(members):,} synthetic members")
    return members


def generate_claims(
    members: pd.DataFrame,
    start_date: date = date(2022, 1, 1),
    end_date:   date = date(2023, 12, 31),
    intervention_effect: float = -420.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic CMS-style claims for a member cohort.

    Args:
        members:              Member DataFrame from generate_members().
        start_date:           First month of claims history.
        end_date:             Last month of claims history.
        intervention_effect:  Annual PMPM cost delta for treated members post-2023.
        seed:                 RNG seed.

    Returns:
        claims DataFrame with one row per claim-line.
    """
    rng = np.random.default_rng(seed + 1)

    # Build list of month-start dates
    service_months = []
    current = start_date.replace(day=1)
    while current <= end_date:
        service_months.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    all_claims = []
    for _, member in members.iterrows():
        all_claims.extend(
            _generate_claims_for_member(member, service_months, intervention_effect, rng)
        )

    claims_df = pd.DataFrame(all_claims)
    logger.info(
        f"Generated {len(claims_df):,} claim lines "
        f"for {len(members):,} members over {len(service_months)} months"
    )
    return claims_df


def generate_and_save(
    output_dir: str,
    n_members:  int   = 10_000,
    n_months:   int   = 24,
    intervention_effect: float = -420.0,
    seed:       int   = 42,
    spark=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full generation pipeline: create members + claims and save as CSV
    to output_dir for Bronze ingestion.

    Returns:
        (members_df, claims_df)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    members = generate_members(n=n_members, seed=seed)

    end_date = date(2022, 1, 1)
    for _ in range(n_months - 1):
        if end_date.month == 12:
            end_date = end_date.replace(year=end_date.year + 1, month=1)
        else:
            end_date = end_date.replace(month=end_date.month + 1)

    claims = generate_claims(
        members=members,
        start_date=date(2022, 1, 1),
        end_date=end_date,
        intervention_effect=intervention_effect,
        seed=seed,
    )

    members_path = f"{output_dir}/members.csv"
    claims_path  = f"{output_dir}/claims.csv"

    members.to_csv(members_path, index=False)
    claims.to_csv(claims_path,  index=False)

    logger.info(f"Saved members → {members_path}")
    logger.info(f"Saved claims  → {claims_path}")

    return members, claims


if __name__ == "__main__":
    # Quick local test
    m, c = generate_and_save(
        output_dir="data/raw",
        n_members=1_000,
        n_months=12,
    )
    print(f"Members: {len(m):,}")
    print(f"Claims:  {len(c):,}")
    print(m.head(3).to_string())
    print(c.head(3).to_string())
