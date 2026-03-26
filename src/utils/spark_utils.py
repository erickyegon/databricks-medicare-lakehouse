"""
src/utils/spark_utils.py
────────────────────────
Shared helpers: SparkSession factory, logging, Delta path resolution,
and schema validation utilities used across all pipeline layers.
"""

import logging
import sys
from datetime import datetime
from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType


# ── Logging ────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── SparkSession ───────────────────────────────────────────────────────────

def get_spark(app_name: str = "MedicareLakehouse") -> SparkSession:
    """
    Return the active SparkSession (Databricks already provides one).
    Falls back to creating a local session for unit tests.
    """
    try:
        spark = SparkSession.getActiveSession()
        if spark is not None:
            return spark
    except Exception:
        pass

    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


# ── Delta helpers ──────────────────────────────────────────────────────────

def table_exists(spark: SparkSession, table_name: str) -> bool:
    """Check if a Delta / Unity Catalog table exists."""
    try:
        spark.sql(f"DESCRIBE TABLE {table_name}")
        return True
    except Exception:
        return False


def get_table_version(spark: SparkSession, table_name: str) -> int:
    """Return the current Delta version for a managed table."""
    hist = spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 1")
    return hist.collect()[0]["version"]


def read_delta_version(
    spark: SparkSession,
    table_name: str,
    version: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> DataFrame:
    """
    Read a Delta table with optional time travel.

    Args:
        table_name: Fully qualified table name.
        version:    Delta version to read.
        timestamp:  Timestamp string for time travel (e.g. "2024-01-01").
    """
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", version)
    elif timestamp is not None:
        reader = reader.option("timestampAsOf", timestamp)
    return reader.table(table_name)


# ── Schema validation ──────────────────────────────────────────────────────

def validate_schema(df: DataFrame, expected_schema: StructType, strict: bool = False) -> List[str]:
    """
    Validate a DataFrame's schema against an expected StructType.

    Args:
        df:              DataFrame to validate.
        expected_schema: Expected schema.
        strict:          If True, raises on any mismatch; otherwise returns warnings.

    Returns:
        List of warning/error messages (empty = clean).
    """
    issues = []
    expected_fields = {f.name: f.dataType for f in expected_schema.fields}
    actual_fields   = {f.name: f.dataType for f in df.schema.fields}

    for col_name, dtype in expected_fields.items():
        if col_name not in actual_fields:
            issues.append(f"MISSING column: {col_name} ({dtype})")
        elif actual_fields[col_name] != dtype:
            issues.append(
                f"TYPE MISMATCH: {col_name} — expected {dtype}, got {actual_fields[col_name]}"
            )

    if strict and issues:
        raise ValueError(f"Schema validation failed:\n" + "\n".join(issues))

    return issues


# ── Data quality helpers ───────────────────────────────────────────────────

def add_audit_columns(df: DataFrame, layer: str) -> DataFrame:
    """
    Add standard audit columns to any DataFrame before writing.

    Columns added:
        _ingested_at   — UTC timestamp of when the row was written
        _pipeline_layer — 'bronze', 'silver', or 'gold'
        _batch_id       — yyyyMMddHHmmss run identifier
    """
    batch_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return (
        df
        .withColumn("_ingested_at",    F.current_timestamp())
        .withColumn("_pipeline_layer", F.lit(layer))
        .withColumn("_batch_id",       F.lit(batch_id))
    )


def count_nulls(df: DataFrame) -> DataFrame:
    """Return a summary DataFrame showing null counts per column."""
    return df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])


def flag_quality_issues(df: DataFrame, rules: dict) -> DataFrame:
    """
    Add a boolean `_quality_pass` column and individual flag columns.

    Args:
        df:    Input DataFrame.
        rules: Dict of {flag_column_name: Column expression (True = issue detected)}.

    Example:
        rules = {
            "flag_negative_amount": F.col("amount") < 0,
            "flag_future_date":     F.col("service_date") > F.current_date(),
        }
    """
    for flag_col, condition in rules.items():
        df = df.withColumn(flag_col, condition)

    flag_cols = list(rules.keys())
    quality_pass = ~(
        F.array(*[F.col(c) for c in flag_cols])
        .cast("array<boolean>")
        .getItem(0)  # placeholder — overridden below
    )
    # Build: pass = none of the flags are True
    combined = F.lit(False)
    for c in flag_cols:
        combined = combined | F.col(c)
    df = df.withColumn("_quality_pass", ~combined)
    return df


# ── Write helpers ──────────────────────────────────────────────────────────

def write_delta(
    df: DataFrame,
    table_name: str,
    mode: str = "overwrite",
    partition_by: Optional[List[str]] = None,
    merge_schema: bool = False,
    optimize_after: bool = False,
    zorder_cols: Optional[List[str]] = None,
    spark: Optional[SparkSession] = None,
) -> int:
    """
    Write a DataFrame to a Delta table with consistent options.

    Returns:
        Row count written.
    """
    logger = get_logger("write_delta")
    row_count = df.count()

    writer = (
        df.write
        .format("delta")
        .mode(mode)
        .option("mergeSchema", str(merge_schema).lower())
    )

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.saveAsTable(table_name)
    logger.info(f"Wrote {row_count:,} rows → {table_name} (mode={mode})")

    if optimize_after and spark is not None:
        zorder = f"ZORDER BY ({', '.join(zorder_cols)})" if zorder_cols else ""
        spark.sql(f"OPTIMIZE {table_name} {zorder}")
        logger.info(f"OPTIMIZE complete: {table_name}")

    return row_count


def upsert_delta(
    spark: SparkSession,
    updates_df: DataFrame,
    target_table: str,
    merge_keys: List[str],
) -> None:
    """
    MERGE (upsert) updates_df into an existing Delta table.
    Creates the table if it does not exist.
    """
    from delta.tables import DeltaTable

    logger = get_logger("upsert_delta")

    if not table_exists(spark, target_table):
        write_delta(updates_df, target_table, mode="overwrite")
        return

    target = DeltaTable.forName(spark, target_table)
    condition = " AND ".join([f"t.{k} = s.{k}" for k in merge_keys])

    (
        target.alias("t")
        .merge(updates_df.alias("s"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info(f"Upsert complete → {target_table} (keys: {merge_keys})")
