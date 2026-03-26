"""
src/ml/drift_monitor.py
────────────────────────
Population Stability Index (PSI) drift monitoring for the Medicare
Claims Lakehouse risk model.

Extracted from risk_model.py into its own module so that:
  - PSI logic is independently unit-testable (no model dependencies)
  - Drift reports can be generated on any two feature store snapshots
    without triggering a full model training run
  - The MLflow logging concern is cleanly separated from the metric
    computation concern

PSI interpretation (healthcare actuarial standard):
  PSI < 0.10    STABLE   — population is consistent with baseline
  PSI 0.10–0.25 WARNING  — moderate shift, investigate before next cycle
  PSI > 0.25    CRITICAL — significant drift, retrain model immediately

Usage:
    from src.ml.drift_monitor import monitor_drift, compute_psi, PSIReport

    report = monitor_drift(
        baseline_table       = "medicare_lakehouse.ml_features.risk_feature_store",
        current_table        = "medicare_lakehouse.ml_features.risk_feature_store_latest",
        features_to_monitor  = DEFAULT_DRIFT_FEATURES,
        spark                = spark,
    )
    print(report.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from config.pipeline_config import CONFIG, MLConfig
from src.utils.spark_utils import get_logger, get_spark

logger = get_logger(__name__)


# ── Default features to monitor each pipeline cycle ───────────────────────
# Covers RAF score, cost, and the most clinically significant utilization
# and HCC burden signals that would indicate population composition shift.

DEFAULT_DRIFT_FEATURES = [
    "raf_score",
    "pre_pmpm",
    "max_hcc_burden",
    "max_hcc_raf",
    "pre_ip_admits",
    "pre_ed_visits",
    "estimated_annual_cost",
    "demographic_raf",
]


# ── PSI result dataclass ───────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    """PSI result for a single feature."""
    feature:    str
    psi:        float
    status:     str          # "STABLE" | "WARNING" | "CRITICAL"
    n_baseline: int
    n_current:  int

    @property
    def is_alert(self) -> bool:
        return self.status in ("WARNING", "CRITICAL")

    def __str__(self) -> str:
        icon = {"STABLE": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(self.status, "?")
        return f"{icon}  {self.feature:<30}  PSI={self.psi:.4f}  [{self.status}]"


@dataclass
class PSIReport:
    """
    Complete drift report across all monitored features.

    Attributes:
        results:          Per-feature FeatureDriftResult list
        baseline_table:   Source table name for baseline
        current_table:    Source table name for current snapshot
        run_timestamp:    When this report was generated (UTC)
    """
    results:          List[FeatureDriftResult]
    baseline_table:   str
    current_table:    str
    run_timestamp:    str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    @property
    def alerts(self) -> List[FeatureDriftResult]:
        return [r for r in self.results if r.is_alert]

    @property
    def critical(self) -> List[FeatureDriftResult]:
        return [r for r in self.results if r.status == "CRITICAL"]

    @property
    def n_features(self) -> int:
        return len(self.results)

    @property
    def n_alerts(self) -> int:
        return len(self.alerts)

    @property
    def max_psi(self) -> float:
        return max((r.psi for r in self.results), default=0.0)

    @property
    def overall_status(self) -> str:
        if any(r.status == "CRITICAL" for r in self.results):
            return "CRITICAL"
        if any(r.status == "WARNING" for r in self.results):
            return "WARNING"
        return "STABLE"

    def summary(self) -> str:
        """Human-readable summary for notebook display or log output."""
        lines = [
            f"PSI Drift Report — {self.run_timestamp}",
            f"Baseline : {self.baseline_table}",
            f"Current  : {self.current_table}",
            f"Overall  : {self.overall_status}  "
            f"({self.n_features} features, {self.n_alerts} alerts, max PSI={self.max_psi:.4f})",
            "",
        ]
        for r in sorted(self.results, key=lambda x: x.psi, reverse=True):
            lines.append(str(r))
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Flat dict for MLflow metric logging."""
        return {
            f"psi_{r.feature}": r.psi
            for r in self.results
        }

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame representation for notebook display."""
        return pd.DataFrame([
            {
                "feature":    r.feature,
                "psi":        r.psi,
                "status":     r.status,
                "n_baseline": r.n_baseline,
                "n_current":  r.n_current,
            }
            for r in sorted(self.results, key=lambda x: x.psi, reverse=True)
        ])


# ── Core PSI computation ───────────────────────────────────────────────────

def compute_psi(
    expected:  pd.Series,
    actual:    pd.Series,
    n_bins:    int = 10,
) -> float:
    """
    Compute Population Stability Index between two univariate distributions.

    Uses percentile-based binning on the expected (baseline) distribution
    so that each bin contains roughly equal baseline mass, which is the
    standard approach in healthcare actuarial and credit risk analytics.

    Args:
        expected:  Baseline distribution (Series of numeric values).
        actual:    Current distribution to compare against baseline.
        n_bins:    Number of bins. 10 is standard; use more for larger samples.

    Returns:
        PSI scalar. Interpretation:
          < 0.10  → STABLE
          0.10–0.25 → WARNING
          > 0.25  → CRITICAL
    """
    if len(expected) == 0 or len(actual) == 0:
        logger.warning("compute_psi received empty series — returning 0.0")
        return 0.0

    # Build bins from baseline percentiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.unique(np.percentile(expected.dropna(), percentiles))

    if len(bins) < 2:
        logger.warning(f"compute_psi: degenerate distribution (all values identical) — returning 0.0")
        return 0.0

    expected_counts = np.histogram(expected.dropna(), bins=bins)[0]
    actual_counts   = np.histogram(actual.dropna(),   bins=bins)[0]

    # Convert to proportions
    expected_pct = expected_counts / max(len(expected.dropna()), 1)
    actual_pct   = actual_counts   / max(len(actual.dropna()),   1)

    # Avoid log(0): replace zeros with small epsilon
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(psi, 4)


def _classify_psi(
    psi:    float,
    config: MLConfig,
) -> str:
    """Map a PSI scalar to a status string using configured thresholds."""
    if psi > config.psi_critical_threshold:
        return "CRITICAL"
    if psi > config.psi_warning_threshold:
        return "WARNING"
    return "STABLE"


# ── Main monitoring function ───────────────────────────────────────────────

def monitor_drift(
    baseline_table:      str,
    current_table:       str,
    features_to_monitor: List[str] = None,
    spark=None,
    config:              Optional[MLConfig] = None,
    log_to_mlflow:       bool = True,
) -> PSIReport:
    """
    Compute PSI across monitored features between a baseline and current
    feature store snapshot. Returns a structured PSIReport.

    When called during an active MLflow run, drift metrics are logged
    automatically as `psi_{feature_name}` scalars.

    Args:
        baseline_table:       Fully qualified Delta table name for baseline.
        current_table:        Fully qualified Delta table name for current data.
        features_to_monitor:  Feature names to check. Defaults to DEFAULT_DRIFT_FEATURES.
        spark:                SparkSession. Uses active session if None.
        config:               MLConfig for PSI thresholds. Uses CONFIG.ml if None.
        log_to_mlflow:        Whether to log PSI metrics to an active MLflow run.

    Returns:
        PSIReport with per-feature results and overall status.
    """
    spark  = spark  or get_spark()
    config = config or CONFIG.ml
    features = features_to_monitor or DEFAULT_DRIFT_FEATURES

    logger.info(
        f"PSI drift monitoring: baseline={baseline_table} | "
        f"current={current_table} | features={len(features)}"
    )

    baseline_df = spark.table(baseline_table).toPandas()
    current_df  = spark.table(current_table).toPandas()

    results: List[FeatureDriftResult] = []

    for feat in features:
        if feat not in baseline_df.columns:
            logger.warning(f"Feature '{feat}' not in baseline table — skipping")
            continue
        if feat not in current_df.columns:
            logger.warning(f"Feature '{feat}' not in current table — skipping")
            continue

        psi    = compute_psi(baseline_df[feat], current_df[feat])
        status = _classify_psi(psi, config)

        result = FeatureDriftResult(
            feature    = feat,
            psi        = psi,
            status     = status,
            n_baseline = len(baseline_df[feat].dropna()),
            n_current  = len(current_df[feat].dropna()),
        )
        results.append(result)
        logger.info(str(result))

    report = PSIReport(
        results         = results,
        baseline_table  = baseline_table,
        current_table   = current_table,
    )

    logger.info(
        f"Drift report complete — overall: {report.overall_status} | "
        f"alerts: {report.n_alerts}/{report.n_features} | max PSI: {report.max_psi:.4f}"
    )

    # Log to active MLflow run if present
    if log_to_mlflow:
        try:
            if mlflow.active_run():
                mlflow.log_metrics(report.to_dict())
                mlflow.log_metric("psi_n_alerts",   report.n_alerts)
                mlflow.log_metric("psi_max",         report.max_psi)
                mlflow.log_param("drift_status",     report.overall_status)
                logger.info(f"PSI metrics logged to MLflow run {mlflow.active_run().info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow logging skipped: {e}")

    return report
