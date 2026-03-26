"""
src/silver/hcc_mapper.py
────────────────────────
ICD-10-CM to HCC v28 mapping for Silver layer enrichment.

Maps raw ICD-10 codes from claims to:
  - HCC category numbers
  - HCC description labels
  - RAF coefficients (community non-dual, aged)
  - Interaction term flags (CHF×AFib, CKD×Diabetes, etc.)

This replicates the CMS HCC v28 mapping logic used in Medicare
Advantage risk adjustment. For production use, replace the embedded
catalogue with the official CMS crosswalk file.
"""

from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType, ArrayType
)

from src.utils.spark_utils import get_logger

logger = get_logger(__name__)


# ── HCC v28 Catalogue ──────────────────────────────────────────────────────
# Source: CMS HCC Model v28 ICD-10-CM Mappings (representative subset)
# Format: icd10_code → (hcc_number, description, raf_coefficient)

HCC_MAPPING: Dict[str, Tuple[int, str, float]] = {
    # Cardiovascular
    "I500":  (85,  "Congestive Heart Failure",            0.323),
    "I501":  (85,  "Congestive Heart Failure",            0.323),
    "I509":  (85,  "Congestive Heart Failure",            0.323),
    "I480":  (96,  "Atrial Fibrillation / Flutter",       0.421),
    "I481":  (96,  "Atrial Fibrillation / Flutter",       0.421),
    "I489":  (96,  "Atrial Fibrillation / Flutter",       0.421),
    "I2109": (86,  "Acute MI",                            0.218),
    "I2110": (86,  "Acute MI",                            0.218),
    "I7001": (107, "Vascular Disease w/ Complications",   0.299),
    "I7011": (107, "Vascular Disease w/ Complications",   0.299),
    "I739":  (108, "Vascular Disease",                    0.178),
    "I731":  (108, "Vascular Disease",                    0.178),

    # Diabetes
    "E1140": (18,  "T2DM with CKD",                       0.302),
    "E1141": (18,  "T2DM with CKD",                       0.302),
    "E1142": (18,  "T2DM with CKD",                       0.302),
    "E119":  (19,  "T2DM Without Complications",           0.118),
    "E118":  (19,  "T2DM Without Complications",           0.118),
    "E1010": (17,  "T1DM Acute Complication",              0.302),
    "E1011": (17,  "T1DM Acute Complication",              0.302),

    # Renal
    "N183":  (138, "CKD Stage 3",                          0.071),
    "N184":  (137, "CKD Stage 4",                          0.138),
    "N185":  (136, "CKD Stage 5",                          0.143),
    "N19":   (135, "Renal Failure",                        0.289),
    "N189":  (135, "Renal Failure",                        0.289),
    "Z992":  (134, "Dialysis Status",                      0.289),
    "Z4901": (134, "Dialysis Status",                      0.289),

    # Pulmonary
    "J449":  (111, "COPD",                                 0.245),
    "J440":  (111, "COPD",                                 0.245),
    "J441":  (111, "COPD",                                 0.245),
    "J84189":(110, "Cystic Fibrosis / ILD",               0.335),

    # Cancer
    "C7951": (8,   "Metastatic Cancer",                    2.488),
    "C7952": (8,   "Metastatic Cancer",                    2.488),
    "C349":  (9,   "Lung / Severe Cancer",                 0.899),
    "C340":  (9,   "Lung / Severe Cancer",                 0.899),
    "C189":  (11,  "Colorectal / Bladder Cancer",          0.439),
    "C180":  (11,  "Colorectal / Bladder Cancer",          0.439),
    "C509":  (12,  "Breast Cancer",                        0.150),
    "C500":  (12,  "Breast Cancer",                        0.150),

    # Neurological
    "G409":  (79,  "Seizure Disorders",                    0.448),
    "G400":  (79,  "Seizure Disorders",                    0.448),
    "G20":   (78,  "Parkinson's Disease",                  0.406),
    "G35":   (77,  "Multiple Sclerosis",                   0.597),
    "M0500": (40,  "Rheumatoid Arthritis",                 0.455),
    "M0510": (40,  "Rheumatoid Arthritis",                 0.455),
    "F329":  (58,  "Major Depression",                     0.421),
    "F330":  (58,  "Major Depression",                     0.421),
    "F209":  (57,  "Schizophrenia",                        0.625),
    "F200":  (57,  "Schizophrenia",                        0.625),

    # Other
    "E6601": (22,  "Morbid Obesity",                       0.178),
    "E6609": (22,  "Morbid Obesity",                       0.178),
}

# Demographic RAF coefficients (age-sex, community non-dual)
DEMOGRAPHIC_RAF: Dict[Tuple[str, int], float] = {
    ("F", 65): 0.321, ("F", 70): 0.382, ("F", 75): 0.453,
    ("F", 80): 0.521, ("F", 85): 0.591, ("F", 90): 0.658, ("F", 95): 0.712,
    ("M", 65): 0.346, ("M", 70): 0.401, ("M", 75): 0.478,
    ("M", 80): 0.549, ("M", 85): 0.619, ("M", 90): 0.685, ("M", 95): 0.740,
}

# HCC interaction terms (CMS v28 specification)
INTERACTION_TERMS = {
    "interaction_chf_afib":     {"hccs": {85, 96},  "coeff": 0.175, "label": "CHF x AFib"},
    "interaction_chf_diabetes": {"hccs": {85, 18, 19}, "coeff": 0.156, "label": "CHF x Diabetes"},
    "interaction_ckd_diabetes": {"hccs": {134, 135, 136, 137, 138, 18, 19}, "coeff": 0.156, "label": "CKD x Diabetes"},
    "interaction_cancer_immune":{"hccs": {8, 9, 11},  "coeff": 0.100, "label": "Cancer x Immune"},
}


# ── PySpark UDF for ICD-10 → HCC mapping ──────────────────────────────────

def _build_hcc_lookup_broadcast(spark: SparkSession):
    """Broadcast the HCC mapping dict to all Spark executors."""
    return spark.sparkContext.broadcast(HCC_MAPPING)


def _icd_to_hcc_udf(icd_codes_pipe_delimited: Optional[str]) -> List[int]:
    """
    Given a pipe-delimited string of ICD-10 codes, return list of HCC numbers.
    Registered as a Spark UDF.
    """
    if not icd_codes_pipe_delimited:
        return []
    codes = icd_codes_pipe_delimited.split("|")
    hccs  = set()
    for code in codes:
        code = code.strip().upper()
        if code in HCC_MAPPING:
            hccs.add(HCC_MAPPING[code][0])
    return sorted(list(hccs))


def _total_hcc_raf_udf(icd_codes_pipe_delimited: Optional[str]) -> float:
    """Sum RAF coefficients for all mapped HCCs from pipe-delimited ICD codes."""
    if not icd_codes_pipe_delimited:
        return 0.0
    codes  = icd_codes_pipe_delimited.split("|")
    seen   = set()
    total  = 0.0
    for code in codes:
        code = code.strip().upper()
        if code in HCC_MAPPING:
            hcc_num = HCC_MAPPING[code][0]
            if hcc_num not in seen:
                total += HCC_MAPPING[code][2]
                seen.add(hcc_num)
    return round(total, 4)


def _primary_hcc_udf(icd10_primary: Optional[str]) -> Optional[int]:
    """Return HCC number for the primary ICD-10 code, or None."""
    if not icd10_primary:
        return None
    code = icd10_primary.strip().upper()
    mapping = HCC_MAPPING.get(code)
    return mapping[0] if mapping else None


def _primary_hcc_desc_udf(icd10_primary: Optional[str]) -> Optional[str]:
    """Return HCC description for the primary ICD-10 code."""
    if not icd10_primary:
        return None
    code = icd10_primary.strip().upper()
    mapping = HCC_MAPPING.get(code)
    return mapping[1] if mapping else "Unknown / Not Mapped"


# ── Main class ─────────────────────────────────────────────────────────────

class HCCMapper:
    """
    Maps ICD-10 codes in Silver claims to HCC categories and RAF scores.

    Steps:
      1. Register UDFs for HCC lookup
      2. Add hcc_list, hcc_raf_total, primary_hcc columns to claims
      3. Add interaction term flags
      4. Output enriched DataFrame

    Usage:
        mapper = HCCMapper(spark)
        enriched_df = mapper.map(clean_claims_df)
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._register_udfs()

    def _register_udfs(self) -> None:
        """Register all HCC-related UDFs with Spark."""
        self.spark.udf.register(
            "icd_to_hcc_list",
            _icd_to_hcc_udf,
            ArrayType(IntegerType())
        )
        self.spark.udf.register(
            "icd_to_hcc_raf_total",
            _total_hcc_raf_udf,
            DoubleType()
        )
        self.spark.udf.register(
            "icd_to_primary_hcc",
            _primary_hcc_udf,
            IntegerType()
        )
        self.spark.udf.register(
            "icd_to_primary_hcc_desc",
            _primary_hcc_desc_udf,
            StringType()
        )
        logger.info("HCC UDFs registered")

    def map(self, df: DataFrame) -> DataFrame:
        """
        Enrich a claims DataFrame with HCC mappings.

        Expected input columns: icd10_codes (pipe-delimited), icd10_primary.

        Added columns:
            hcc_list          — array of HCC numbers from all ICD-10 codes
            hcc_raf_total     — sum of RAF coefficients for mapped HCCs
            primary_hcc       — HCC number of the primary diagnosis
            primary_hcc_desc  — Description of primary HCC
            hcc_burden_count  — number of distinct HCCs on this claim
            has_chf / has_afib / has_diabetes / has_ckd / has_cancer
                              — boolean flags for key condition groups
        """
        logger.info("Applying HCC mapping to claims")

        icd_to_hcc_list_udf        = F.udf(_icd_to_hcc_udf,        ArrayType(IntegerType()))
        icd_to_hcc_raf_total_udf   = F.udf(_total_hcc_raf_udf,     DoubleType())
        icd_to_primary_hcc_udf     = F.udf(_primary_hcc_udf,       IntegerType())
        icd_to_primary_hcc_desc_udf= F.udf(_primary_hcc_desc_udf,  StringType())

        df = (
            df
            .withColumn("hcc_list",         icd_to_hcc_list_udf(F.col("icd10_codes")))
            .withColumn("hcc_raf_total",     icd_to_hcc_raf_total_udf(F.col("icd10_codes")))
            .withColumn("primary_hcc",       icd_to_primary_hcc_udf(F.col("icd10_primary")))
            .withColumn("primary_hcc_desc",  icd_to_primary_hcc_desc_udf(F.col("icd10_primary")))
            .withColumn("hcc_burden_count",  F.size("hcc_list"))
        )

        # Condition group flags (used for interaction terms and feature engineering)
        df = (
            df
            .withColumn("has_chf",      F.array_contains("hcc_list", 85))
            .withColumn("has_afib",     F.array_contains("hcc_list", 96))
            .withColumn("has_diabetes", (
                F.array_contains("hcc_list", 17) |
                F.array_contains("hcc_list", 18) |
                F.array_contains("hcc_list", 19)
            ))
            .withColumn("has_ckd", (
                F.array_contains("hcc_list", 134) |
                F.array_contains("hcc_list", 135) |
                F.array_contains("hcc_list", 136) |
                F.array_contains("hcc_list", 137) |
                F.array_contains("hcc_list", 138)
            ))
            .withColumn("has_cancer", (
                F.array_contains("hcc_list", 8)  |
                F.array_contains("hcc_list", 9)  |
                F.array_contains("hcc_list", 11) |
                F.array_contains("hcc_list", 12)
            ))
            .withColumn("has_copd",         F.array_contains("hcc_list", 111))
            .withColumn("has_depression",   F.array_contains("hcc_list", 58))
            .withColumn("has_metastatic",   F.array_contains("hcc_list", 8))
        )

        # Interaction terms (additive RAF coefficients)
        df = (
            df
            .withColumn("interaction_chf_afib",
                (F.col("has_chf") & F.col("has_afib")).cast(DoubleType()) * 0.175)
            .withColumn("interaction_chf_diabetes",
                (F.col("has_chf") & F.col("has_diabetes")).cast(DoubleType()) * 0.156)
            .withColumn("interaction_ckd_diabetes",
                (F.col("has_ckd") & F.col("has_diabetes")).cast(DoubleType()) * 0.156)
            .withColumn("hcc_interaction_total",
                F.col("interaction_chf_afib") +
                F.col("interaction_chf_diabetes") +
                F.col("interaction_ckd_diabetes")
            )
        )

        logger.info(f"HCC mapping complete — {df.count():,} claim rows enriched")
        return df


def get_demographic_raf(sex: str, age: int) -> float:
    """Look up demographic RAF coefficient for age-sex cell."""
    age_bracket = min(range(65, 100, 5), key=lambda a: abs(a - max(65, min(age, 95))))
    return DEMOGRAPHIC_RAF.get((sex, age_bracket), 0.45)


def build_hcc_reference_df(spark: SparkSession) -> DataFrame:
    """Return a Spark DataFrame of the full HCC mapping catalogue."""
    rows = [
        (code, hcc, desc, coeff)
        for code, (hcc, desc, coeff) in HCC_MAPPING.items()
    ]
    schema = StructType([
        StructField("icd10_code",      StringType(),  False),
        StructField("hcc_number",      IntegerType(), True),
        StructField("hcc_description", StringType(),  True),
        StructField("raf_coefficient", DoubleType(),  True),
    ])
    return spark.createDataFrame(rows, schema)
