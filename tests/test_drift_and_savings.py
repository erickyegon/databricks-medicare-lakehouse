"""
tests/test_drift_and_savings.py
────────────────────────────────
Unit tests for:
  - src/ml/drift_monitor.py   (PSI computation, PSIReport, monitor thresholds)
  - src/gold/shared_savings.py (SharedSavingsCalculator, SharedSavingsResult)

Run with: pytest tests/test_drift_and_savings.py -v
These tests require only pandas and numpy — no Spark or MLflow needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

# ── Shared savings tests ───────────────────────────────────────────────────

from src.gold.shared_savings import SharedSavingsCalculator, SharedSavingsResult


class TestSharedSavingsResult:

    def test_gross_savings_positive_when_actual_below_benchmark(self):
        r = SharedSavingsResult(
            benchmark_pmpm=816.67, actual_pmpm=784.12,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        assert r.gross_savings > 0

    def test_gross_savings_negative_when_actual_above_benchmark(self):
        r = SharedSavingsResult(
            benchmark_pmpm=816.67, actual_pmpm=900.0,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        assert r.gross_savings < 0

    def test_earned_zero_when_below_msr(self):
        # Savings rate ~0.5% — below 2% MSR
        r = SharedSavingsResult(
            benchmark_pmpm=1000.0, actual_pmpm=995.0,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        assert not r.exceeds_msr
        assert r.earned_savings == 0.0
        assert r.status == "BELOW_MSR"

    def test_earned_nonzero_when_above_msr(self):
        # Savings rate ~4% — above 2% MSR
        r = SharedSavingsResult(
            benchmark_pmpm=1000.0, actual_pmpm=960.0,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        assert r.exceeds_msr
        assert r.earned_savings > 0
        assert r.status == "SAVINGS_EARNED"

    def test_earned_equals_gross_times_sharing_rate(self):
        r = SharedSavingsResult(
            benchmark_pmpm=1000.0, actual_pmpm=950.0,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        assert abs(r.earned_savings - r.gross_savings * 0.50) < 0.01

    def test_per_member_gross_formula(self):
        r = SharedSavingsResult(
            benchmark_pmpm=1000.0, actual_pmpm=950.0,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        expected_gross = (1000.0 - 950.0) * 10_000 * 12
        assert abs(r.gross_savings - expected_gross) < 1.0
        assert abs(r.per_member_gross - expected_gross / 10_000) < 0.01

    def test_savings_rate_calculation(self):
        r = SharedSavingsResult(
            benchmark_pmpm=1000.0, actual_pmpm=960.0,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        # (1000-960)/1000 = 4%
        assert abs(r.savings_rate - 0.04) < 0.001

    def test_to_dict_contains_required_keys(self):
        r = SharedSavingsResult(
            benchmark_pmpm=816.67, actual_pmpm=784.12,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        d = r.to_dict()
        required = [
            "benchmark_pmpm", "actual_pmpm", "n_lives", "gross_savings",
            "savings_rate", "earned_savings", "exceeds_msr", "status"
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_str_output_contains_dollar_signs(self):
        r = SharedSavingsResult(
            benchmark_pmpm=816.67, actual_pmpm=784.12,
            n_lives=10_000, sharing_rate=0.50, msr=0.02
        )
        s = str(r)
        assert "$" in s
        assert "SAVINGS_EARNED" in s or "BELOW_MSR" in s


class TestSharedSavingsCalculator:

    @pytest.fixture
    def calc(self):
        return SharedSavingsCalculator()

    def test_compute_returns_result(self, calc):
        result = calc.compute(816.67, 784.12, 10_000)
        assert isinstance(result, SharedSavingsResult)

    def test_from_att_391_exceeds_msr(self, calc):
        # ATT = -$391/member/year should exceed 2% MSR on standard benchmark
        result = calc.from_att(att_pmpm_annual=391.0, n_lives=10_000)
        assert result.exceeds_msr, (
            f"ATT=$391 should exceed MSR but savings_rate={result.savings_rate:.2%}"
        )

    def test_from_att_zero_no_savings(self, calc):
        result = calc.from_att(att_pmpm_annual=0.0, n_lives=10_000)
        assert result.gross_savings == pytest.approx(0.0, abs=1.0)
        assert not result.exceeds_msr

    def test_from_att_annualised_correctly(self, calc):
        # att_pmpm_annual=120 → monthly saving = $10 → actual_pmpm = benchmark - 10
        result = calc.from_att(att_pmpm_annual=120.0, n_lives=1_000)
        expected_gross = 120.0 * 1_000
        assert abs(result.gross_savings - expected_gross) < 1.0

    def test_project_returns_dataframe(self, calc):
        df = calc.project(att_pmpm=391.0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(SharedSavingsCalculator.REFERENCE_POPULATIONS)

    def test_project_columns_present(self, calc):
        df = calc.project(att_pmpm=391.0)
        for col in ["Gross Savings", "Savings Rate", "Earned (50% SR)", "MSR Exceeded"]:
            assert col in df.columns

    def test_larger_population_earns_proportionally_more(self, calc):
        small = calc.from_att(391.0, n_lives=10_000)
        large = calc.from_att(391.0, n_lives=100_000)
        assert large.gross_savings == pytest.approx(small.gross_savings * 10, rel=0.01)

    def test_sharing_rate_override(self, calc):
        r30 = calc.from_att(391.0, n_lives=10_000, sharing_rate=0.30)
        r70 = calc.from_att(391.0, n_lives=10_000, sharing_rate=0.70)
        assert r70.earned_savings > r30.earned_savings
        assert abs(r70.earned_savings / r30.earned_savings - 70/30) < 0.01


# ── Drift monitor tests ────────────────────────────────────────────────────

from src.ml.drift_monitor import (
    compute_psi, _classify_psi, FeatureDriftResult, PSIReport,
    DEFAULT_DRIFT_FEATURES,
)
from config.pipeline_config import CONFIG


class TestComputePSI:

    def test_identical_distributions_return_zero(self):
        s = pd.Series(np.random.default_rng(42).normal(0, 1, 1000))
        assert compute_psi(s, s) == pytest.approx(0.0, abs=0.001)

    def test_shifted_distribution_returns_positive_psi(self):
        rng = np.random.default_rng(42)
        baseline = pd.Series(rng.normal(0, 1, 2000))
        current  = pd.Series(rng.normal(2, 1, 2000))   # shifted mean
        psi = compute_psi(baseline, current)
        assert psi > 0.10, f"Strongly shifted distribution should have PSI > 0.10, got {psi}"

    def test_psi_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            a = pd.Series(rng.uniform(0, 10, 500))
            b = pd.Series(rng.uniform(0, 10, 500))
            assert compute_psi(a, b) >= 0

    def test_empty_series_returns_zero(self):
        assert compute_psi(pd.Series([], dtype=float), pd.Series([1.0, 2.0])) == 0.0

    def test_degenerate_constant_returns_zero(self):
        constant = pd.Series([5.0] * 100)
        current  = pd.Series(np.random.default_rng(1).normal(5, 1, 100))
        # Should not raise — returns 0.0 for degenerate baseline
        result = compute_psi(constant, current)
        assert result == 0.0

    def test_stable_range_threshold(self):
        rng = np.random.default_rng(99)
        baseline = pd.Series(rng.normal(0, 1, 5000))
        current  = pd.Series(rng.normal(0.05, 1, 5000))  # tiny shift
        psi = compute_psi(baseline, current)
        assert psi < 0.10, f"Tiny shift should be STABLE (< 0.10), got {psi}"

    def test_result_is_float(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        t = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5] * 100)
        result = compute_psi(s, t)
        assert isinstance(result, float)


class TestClassifyPSI:

    def test_stable_below_warning(self):
        assert _classify_psi(0.05, CONFIG.ml) == "STABLE"

    def test_warning_between_thresholds(self):
        assert _classify_psi(0.15, CONFIG.ml) == "WARNING"

    def test_critical_above_threshold(self):
        assert _classify_psi(0.30, CONFIG.ml) == "CRITICAL"

    def test_exact_warning_threshold_is_warning(self):
        # PSI exactly at 0.10 (warning threshold) should be WARNING not STABLE
        assert _classify_psi(0.101, CONFIG.ml) == "WARNING"

    def test_exact_critical_threshold_is_critical(self):
        assert _classify_psi(0.251, CONFIG.ml) == "CRITICAL"


class TestFeatureDriftResult:

    def test_is_alert_true_for_warning(self):
        r = FeatureDriftResult("raf_score", 0.15, "WARNING", 1000, 1000)
        assert r.is_alert

    def test_is_alert_true_for_critical(self):
        r = FeatureDriftResult("pre_pmpm", 0.30, "CRITICAL", 1000, 1000)
        assert r.is_alert

    def test_is_alert_false_for_stable(self):
        r = FeatureDriftResult("max_hcc_burden", 0.05, "STABLE", 1000, 1000)
        assert not r.is_alert

    def test_str_contains_feature_name(self):
        r = FeatureDriftResult("raf_score", 0.15, "WARNING", 1000, 1000)
        assert "raf_score" in str(r)


class TestPSIReport:

    @pytest.fixture
    def sample_report(self):
        results = [
            FeatureDriftResult("raf_score",     0.05, "STABLE",   1000, 1000),
            FeatureDriftResult("pre_pmpm",       0.15, "WARNING",  1000, 1000),
            FeatureDriftResult("max_hcc_burden", 0.30, "CRITICAL", 1000, 1000),
        ]
        return PSIReport(results, "baseline_table", "current_table")

    def test_overall_status_critical_when_any_critical(self, sample_report):
        assert sample_report.overall_status == "CRITICAL"

    def test_n_alerts_counts_warning_and_critical(self, sample_report):
        assert sample_report.n_alerts == 2

    def test_max_psi_correct(self, sample_report):
        assert sample_report.max_psi == pytest.approx(0.30)

    def test_critical_property_returns_only_critical(self, sample_report):
        assert len(sample_report.critical) == 1
        assert sample_report.critical[0].feature == "max_hcc_burden"

    def test_to_dict_keys_prefixed_with_psi(self, sample_report):
        d = sample_report.to_dict()
        for k in d:
            assert k.startswith("psi_"), f"Key should start with 'psi_': {k}"

    def test_to_dataframe_has_correct_columns(self, sample_report):
        df = sample_report.to_dataframe()
        for col in ["feature", "psi", "status", "n_baseline", "n_current"]:
            assert col in df.columns

    def test_summary_contains_overall_status(self, sample_report):
        s = sample_report.summary()
        assert "CRITICAL" in s

    def test_all_stable_report_is_stable(self):
        results = [
            FeatureDriftResult("a", 0.02, "STABLE", 100, 100),
            FeatureDriftResult("b", 0.05, "STABLE", 100, 100),
        ]
        report = PSIReport(results, "t1", "t2")
        assert report.overall_status == "STABLE"

    def test_default_drift_features_non_empty(self):
        assert len(DEFAULT_DRIFT_FEATURES) >= 5
        assert "raf_score" in DEFAULT_DRIFT_FEATURES
        assert "pre_pmpm"  in DEFAULT_DRIFT_FEATURES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
