"""
src/gold/shared_savings.py
──────────────────────────
MSSP Shared Savings Programme calculator for the Medicare Claims Lakehouse.

Extracted into its own module so that:
  - Savings logic is independently unit-testable
  - Notebook 03 and Gold aggregation call a clean, versioned API
  - Scenario projections can be generated without re-running the full pipeline

Business rules implemented:
  - Gross savings = (benchmark_pmpm - actual_pmpm) × n_lives × 12
  - Savings rate  = gross_savings / (benchmark_pmpm × n_lives × 12)
  - ACO earns     = gross_savings × sharing_rate  IF savings_rate > MSR
  - ACO earns 0   if savings_rate ≤ MSR (minimum savings rate not met)

Reference: CMS MSSP Final Rule (42 CFR Part 425)

Usage:
    from src.gold.shared_savings import SharedSavingsCalculator, SharedSavingsResult

    calc   = SharedSavingsCalculator()
    result = calc.compute(benchmark_pmpm=816.67, actual_pmpm=784.12, n_lives=10000)
    print(result)

    # Projection table across population sizes
    projection = calc.project(att_pmpm=391.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from config.pipeline_config import CONFIG, GoldConfig
from src.utils.spark_utils import get_logger

logger = get_logger(__name__)


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class SharedSavingsResult:
    """
    Structured output from a single SharedSavingsCalculator.compute() call.

    All dollar values are annual totals unless noted as PMPM.
    """
    # Inputs (stored for audit trail)
    benchmark_pmpm:   float
    actual_pmpm:      float
    n_lives:          int
    sharing_rate:     float
    msr:              float

    # Computed outputs
    gross_savings:    float = field(init=False)
    savings_rate:     float = field(init=False)
    earned_savings:   float = field(init=False)
    per_member_gross: float = field(init=False)
    per_member_earned:float = field(init=False)
    exceeds_msr:      bool  = field(init=False)
    status:           str   = field(init=False)

    def __post_init__(self):
        total_benchmark = self.benchmark_pmpm * self.n_lives * 12
        total_actual    = self.actual_pmpm    * self.n_lives * 12

        self.gross_savings     = total_benchmark - total_actual
        self.savings_rate      = self.gross_savings / max(total_benchmark, 1)
        self.exceeds_msr       = self.savings_rate > self.msr
        self.earned_savings    = self.gross_savings * self.sharing_rate if self.exceeds_msr else 0.0
        self.per_member_gross  = self.gross_savings  / max(self.n_lives, 1)
        self.per_member_earned = self.earned_savings / max(self.n_lives, 1)
        self.status = "SAVINGS_EARNED" if self.exceeds_msr else "BELOW_MSR"

    def __str__(self) -> str:
        msr_flag = "✅ exceeds MSR" if self.exceeds_msr else f"❌ below MSR ({self.msr:.1%})"
        return (
            f"SharedSavingsResult\n"
            f"  Benchmark PMPM    : ${self.benchmark_pmpm:>10,.2f}\n"
            f"  Actual PMPM       : ${self.actual_pmpm:>10,.2f}\n"
            f"  N attributed lives: {self.n_lives:>10,}\n"
            f"  ─────────────────────────────────────\n"
            f"  Gross savings     : ${self.gross_savings:>10,.0f}\n"
            f"  Savings rate      : {self.savings_rate:>10.2%}  {msr_flag}\n"
            f"  Earned (at {self.sharing_rate:.0%}): ${self.earned_savings:>10,.0f}\n"
            f"  Per-member gross  : ${self.per_member_gross:>10,.0f}/member/year\n"
            f"  Per-member earned : ${self.per_member_earned:>10,.0f}/member/year\n"
        )

    def to_dict(self) -> dict:
        """Flat dict for MLflow logging or DataFrame row."""
        return {
            "benchmark_pmpm":    self.benchmark_pmpm,
            "actual_pmpm":       self.actual_pmpm,
            "n_lives":           self.n_lives,
            "sharing_rate":      self.sharing_rate,
            "msr":               self.msr,
            "gross_savings":     self.gross_savings,
            "savings_rate":      self.savings_rate,
            "earned_savings":    self.earned_savings,
            "per_member_gross":  self.per_member_gross,
            "per_member_earned": self.per_member_earned,
            "exceeds_msr":       self.exceeds_msr,
            "status":            self.status,
        }


# ── Calculator ──────────────────────────────────────────────────────────────

class SharedSavingsCalculator:
    """
    MSSP Shared Savings Programme calculator.

    Implements CMS MSSP economics:
      1. Gross savings from benchmark vs actual PMPM
      2. Minimum Savings Rate (MSR) gating
      3. Earned savings at specified sharing rate
      4. Scale projections across population scenarios

    Simplifications (documented in README):
      - No risk corridor adjustments
      - No CMS Star rating multipliers
      - No benchmark rebasing after Year 3
      - No regional adjustment factors
      - One-sided risk only (savings model, not risk-sharing model)

    Usage:
        calc = SharedSavingsCalculator()

        # From pipeline actuals
        result = calc.compute(
            benchmark_pmpm = 816.67,
            actual_pmpm    = 784.12,
            n_lives        = 10_000
        )

        # From ATT estimate (DiD output)
        result = calc.from_att(att_pmpm_annual=391.0, n_lives=10_000)

        # Scale projection
        df = calc.project(att_pmpm=391.0)
    """

    # Reference population scenarios for scale projections
    REFERENCE_POPULATIONS: List[Tuple[str, int]] = [
        ("Regional ACO (10K members)",         10_000),
        ("Mid-size MA plan (100K members)",    100_000),
        ("Large MA plan (500K members)",       500_000),
        ("National-scale (1M members)",      1_000_000),
        ("National-scale (4.7M members)",    4_700_000),
    ]

    def __init__(self, config: Optional[GoldConfig] = None):
        self.config = config or CONFIG.gold

    def compute(
        self,
        benchmark_pmpm: float,
        actual_pmpm:    float,
        n_lives:        int,
        sharing_rate:   Optional[float] = None,
        msr:            Optional[float] = None,
    ) -> SharedSavingsResult:
        """
        Compute shared savings from benchmark and actual PMPM.

        Args:
            benchmark_pmpm:  CMS-set expected cost per member per month.
            actual_pmpm:     Observed PMPM from claims (Gold layer output).
            n_lives:         Number of attributed beneficiaries.
            sharing_rate:    ACO's share of gross savings (default: config.sharing_rate).
            msr:             Minimum savings rate to earn any share (default: config.msr_threshold).

        Returns:
            SharedSavingsResult with gross savings, earned savings, and status.
        """
        sr  = sharing_rate if sharing_rate is not None else self.config.sharing_rate
        msr_val = msr if msr is not None else self.config.msr_threshold

        result = SharedSavingsResult(
            benchmark_pmpm = benchmark_pmpm,
            actual_pmpm    = actual_pmpm,
            n_lives        = n_lives,
            sharing_rate   = sr,
            msr            = msr_val,
        )

        logger.info(
            f"Savings computed — gross: ${result.gross_savings:,.0f} | "
            f"rate: {result.savings_rate:.2%} | status: {result.status}"
        )
        return result

    def from_att(
        self,
        att_pmpm_annual: float,
        n_lives:         int,
        sharing_rate:    Optional[float] = None,
        msr:             Optional[float] = None,
    ) -> SharedSavingsResult:
        """
        Compute savings from a DiD ATT estimate.

        The ATT (Average Treatment effect on the Treated) from the
        Difference-in-Differences analysis directly represents the
        annual cost reduction per member. This method converts it
        to MSSP shared savings terms.

        Args:
            att_pmpm_annual:  Annual ATT per member (positive = savings,
                              e.g. pass 391.0 for −$391 cost reduction).
            n_lives:          Number of attributed beneficiaries.

        Returns:
            SharedSavingsResult using benchmark from config.
        """
        benchmark_pmpm = self.config.benchmark_pmpm
        actual_pmpm    = benchmark_pmpm - (att_pmpm_annual / 12)

        return self.compute(
            benchmark_pmpm = benchmark_pmpm,
            actual_pmpm    = actual_pmpm,
            n_lives        = n_lives,
            sharing_rate   = sharing_rate,
            msr            = msr,
        )

    def project(
        self,
        att_pmpm:       float,
        sharing_rate:   Optional[float] = None,
        msr:            Optional[float] = None,
        populations:    Optional[List[Tuple[str, int]]] = None,
    ) -> pd.DataFrame:
        """
        Generate a scale projection table across reference population sizes.

        Args:
            att_pmpm:     Annual ATT per member (positive dollars).
            sharing_rate: ACO sharing rate override.
            msr:          MSR override.
            populations:  List of (label, n_lives) tuples. Defaults to
                          REFERENCE_POPULATIONS.

        Returns:
            DataFrame with one row per population scenario.
        """
        pops = populations or self.REFERENCE_POPULATIONS
        rows = []

        for label, n_lives in pops:
            result = self.from_att(att_pmpm, n_lives, sharing_rate, msr)
            row = {"Scenario": label, "Attributed Lives": f"{n_lives:,}"}
            row["Gross Savings"]    = f"${result.gross_savings:,.0f}"
            row["Savings Rate"]     = f"{result.savings_rate:.1%}"
            row["Earned (50% SR)"]  = f"${result.earned_savings:,.0f}"
            row["Per-Member Gross"] = f"${result.per_member_gross:,.0f}"
            row["MSR Exceeded"]     = "✅" if result.exceeds_msr else "❌"
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Scenario")
        logger.info(f"Projection table generated for {len(pops)} population scenarios")
        return df

    def compute_from_gold(self, spark, n_lives: Optional[int] = None) -> SharedSavingsResult:
        """
        Compute shared savings directly from the Gold member RAF scores table.
        Reads actual_pmpm directly from gold.member_raf_scores.

        Args:
            spark:    SparkSession.
            n_lives:  Override attributed life count. Defaults to row count.

        Returns:
            SharedSavingsResult.
        """
        from pyspark.sql import functions as F

        raf_df = spark.table(self.config.raf_scores_table)

        stats = raf_df.agg(
            F.mean("actual_pmpm").alias("mean_actual_pmpm"),
            F.count("bene_id").alias("n_members"),
        ).collect()[0]

        actual_pmpm = float(stats["mean_actual_pmpm"] or 0)
        n            = n_lives or int(stats["n_members"])

        return self.compute(
            benchmark_pmpm = self.config.benchmark_pmpm,
            actual_pmpm    = actual_pmpm,
            n_lives        = n,
        )
