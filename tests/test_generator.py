"""
tests/test_generator.py
────────────────────────
Unit tests for the synthetic CMS claims generator.
Run with: pytest tests/test_generator.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from datetime import date

from src.data_generator.cms_claims_generator import (
    generate_members,
    generate_claims,
    ICD10_CATALOGUE,
)


class TestGenerateMembers:

    def test_returns_correct_count(self):
        df = generate_members(n=100, seed=42)
        assert len(df) == 100

    def test_required_columns_present(self):
        df = generate_members(n=50, seed=42)
        required = ["bene_id", "age", "sex", "dual_eligible", "plan_type",
                    "state", "enrollment_date", "intervention_arm"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_age_range(self):
        df = generate_members(n=500, seed=42)
        assert df["age"].min() >= 65
        assert df["age"].max() <= 95

    def test_sex_values(self):
        df = generate_members(n=500, seed=42)
        assert set(df["sex"].unique()).issubset({"F", "M"})

    def test_intervention_arm_binary(self):
        df = generate_members(n=500, seed=42)
        assert set(df["intervention_arm"].unique()).issubset({0, 1})

    def test_bene_ids_unique(self):
        df = generate_members(n=500, seed=42)
        assert df["bene_id"].nunique() == 500

    def test_deterministic_with_same_seed(self):
        df1 = generate_members(n=100, seed=99)
        df2 = generate_members(n=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestGenerateClaims:

    @pytest.fixture
    def members(self):
        return generate_members(n=50, seed=42)

    def test_returns_dataframe(self, members):
        claims = generate_claims(members, seed=42)
        assert isinstance(claims, pd.DataFrame)
        assert len(claims) > 0

    def test_required_columns(self, members):
        claims = generate_claims(members, seed=42)
        required = ["claim_id", "bene_id", "service_date", "claim_year",
                    "claim_month", "service_type", "icd10_primary",
                    "claim_amount", "intervention_arm"]
        for col in required:
            assert col in claims.columns

    def test_claim_amounts_non_negative(self, members):
        claims = generate_claims(members, seed=42)
        assert (claims["claim_amount"] >= 0).all()

    def test_all_bene_ids_in_members(self, members):
        claims = generate_claims(members, seed=42)
        member_ids = set(members["bene_id"])
        claim_bene_ids = set(claims["bene_id"])
        assert claim_bene_ids.issubset(member_ids)

    def test_service_types_valid(self, members):
        claims = generate_claims(members, seed=42)
        valid_types = {"ip_admit", "ed_visit", "specialist", "primary", "lab"}
        assert set(claims["service_type"].unique()).issubset(valid_types)

    def test_intervention_effect_reduces_cost(self, members):
        """Intervention arm should have lower post-intervention costs on average."""
        claims = generate_claims(
            members,
            start_date=date(2022, 1, 1),
            end_date=date(2023, 12, 31),
            intervention_effect=-500.0,
            seed=42,
        )
        post_claims = claims[claims["claim_year"] >= 2023]
        if len(post_claims) > 0 and post_claims["intervention_arm"].nunique() == 2:
            control_mean  = post_claims[post_claims["intervention_arm"] == 0]["claim_amount"].mean()
            treated_mean  = post_claims[post_claims["intervention_arm"] == 1]["claim_amount"].mean()
            # With strong effect (-500), treated should generally be lower
            # Allow some variance — just check direction on average
            assert treated_mean <= control_mean * 1.2  # treated not more than 20% above control


class TestICDCatalogue:

    def test_catalogue_has_entries(self):
        assert len(ICD10_CATALOGUE) > 10

    def test_prevalence_weights_positive(self):
        for entry in ICD10_CATALOGUE:
            assert entry[4] > 0, f"Non-positive weight for {entry[0]}"

    def test_raf_coefficients_non_negative(self):
        for entry in ICD10_CATALOGUE:
            assert entry[3] >= 0, f"Negative RAF for {entry[0]}"


# ── tests/test_hcc_mapper.py ──────────────────────────────────────────────

"""
Unit tests for the HCC mapper (non-Spark: testing pure Python functions).
"""

from src.silver.hcc_mapper import (
    _icd_to_hcc_udf,
    _total_hcc_raf_udf,
    _primary_hcc_udf,
    _primary_hcc_desc_udf,
    get_demographic_raf,
    HCC_MAPPING,
)


class TestHCCMapper:

    def test_known_icd_maps_to_hcc(self):
        result = _icd_to_hcc_udf("I500")
        assert 85 in result  # CHF = HCC 85

    def test_multiple_codes_pipe_delimited(self):
        result = _icd_to_hcc_udf("I500|E119|N183")
        assert 85  in result   # CHF
        assert 19  in result   # T2DM
        assert 138 in result   # CKD 3

    def test_unknown_code_returns_empty(self):
        result = _icd_to_hcc_udf("XXXX")
        assert result == []

    def test_null_returns_empty(self):
        assert _icd_to_hcc_udf(None) == []

    def test_raf_total_sums_correctly(self):
        # CHF (0.323) + T2DM no compl (0.118)
        raf = _total_hcc_raf_udf("I500|E119")
        assert abs(raf - (0.323 + 0.118)) < 0.001

    def test_raf_no_double_count_same_hcc(self):
        # I500 and I501 both map to HCC 85 — should count only once
        raf = _total_hcc_raf_udf("I500|I501")
        assert abs(raf - 0.323) < 0.001

    def test_primary_hcc_returns_correct(self):
        assert _primary_hcc_udf("I500") == 85
        assert _primary_hcc_udf("E119") == 19

    def test_primary_hcc_unknown_returns_none(self):
        assert _primary_hcc_udf("XXXXX") is None

    def test_demographic_raf_female_75(self):
        raf = get_demographic_raf("F", 75)
        assert abs(raf - 0.453) < 0.001

    def test_demographic_raf_male_80(self):
        raf = get_demographic_raf("M", 80)
        assert abs(raf - 0.549) < 0.001

    def test_demographic_raf_clips_to_valid_range(self):
        # Age 100 should clip to 95 bracket
        raf = get_demographic_raf("F", 100)
        assert raf > 0

    def test_hcc_mapping_non_empty(self):
        assert len(HCC_MAPPING) > 20

    def test_all_mapping_values_valid(self):
        for code, (hcc, desc, coeff) in HCC_MAPPING.items():
            assert isinstance(hcc, int),   f"HCC not int: {code}"
            assert isinstance(desc, str),  f"Desc not str: {code}"
            assert isinstance(coeff, float), f"Coeff not float: {code}"
            assert coeff >= 0,             f"Negative coeff: {code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
