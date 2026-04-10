# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: Clean and Resample onto SoC Grid
# MAGIC
# MAGIC Reads `silver.session_timeseries`, applies quality filters, and resamples
# MAGIC each session's Power-vs-SoC curve onto a uniform 1% SoC grid.
# MAGIC
# MAGIC Produces: `silver.session_soc_curves` — one row per session with
# MAGIC `soc_grid` and `power_grid` arrays.

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

TARGET_CATALOG = "jonathan_play"
TARGET_SCHEMA = "vehicle_charge_curves"

# COMMAND ----------

# MAGIC %md ## 1. Load Session Timeseries

# COMMAND ----------

timeseries = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_session_timeseries")

# COMMAND ----------

# MAGIC %md ## 2. Define Utility Functions and Pandas UDF
# MAGIC
# MAGIC Functions are defined inline so they're available on worker nodes
# MAGIC when executed via `applyInPandas`.

# COMMAND ----------

def resample_to_soc_grid(soc, power, grid_points=101):
    """Resample a session's power curve onto a uniform SoC grid."""
    soc_grid = np.linspace(soc.min(), soc.max(), grid_points)
    power_grid = np.interp(soc_grid, soc, power)
    return soc_grid, power_grid


def filter_session(soc, power, min_points=5, max_gap_pct=0.20):
    """Check whether a session has sufficient data quality."""
    if len(soc) < min_points:
        return False
    if len(soc) != len(power):
        return False
    soc_range = soc.max() - soc.min()
    if soc_range <= 0:
        return False
    sorted_soc = np.sort(soc)
    gaps = np.diff(sorted_soc)
    if len(gaps) > 0 and gaps.max() / soc_range > max_gap_pct:
        return False
    return True

# COMMAND ----------

output_schema = StructType([
    StructField("session_id", LongType(), False),
    StructField("make", StringType(), True),
    StructField("model", StringType(), True),
    StructField("evse_max_power_kw", FloatType(), True),
    StructField("soc_start", FloatType(), False),
    StructField("soc_end", FloatType(), False),
    StructField("soc_grid", ArrayType(FloatType()), False),
    StructField("power_grid", ArrayType(FloatType()), False),
    StructField("n_raw_points", IntegerType(), False),
    StructField("session_quality_flag", StringType(), False),
])


def resample_session(pdf: pd.DataFrame) -> pd.DataFrame:
    """Process a single session's timeseries into a resampled SoC curve."""
    # Sort by SoC (should be monotonically increasing during charging)
    pdf = pdf.sort_values("soc").dropna(subset=["soc", "power_kw"])

    soc = pdf["soc"].values
    power = pdf["power_kw"].values
    n_raw = len(soc)

    # Classify quality
    if n_raw < 5:
        quality = "sparse"
    elif not filter_session(soc, power, min_points=5, max_gap_pct=0.20):
        quality = "noisy"
    else:
        quality = "good"

    # Only resample sessions with enough data
    if n_raw < 3:
        return pd.DataFrame()

    soc_grid, power_grid = resample_to_soc_grid(soc, power, grid_points=101)

    row = pdf.iloc[0]
    return pd.DataFrame([{
        "session_id": row["session_id"],
        "make": row.get("make"),
        "model": row.get("model"),
        "evse_max_power_kw": row.get("evse_max_power_kw"),
        "soc_start": float(soc.min()),
        "soc_end": float(soc.max()),
        "soc_grid": soc_grid.tolist(),
        "power_grid": power_grid.tolist(),
        "n_raw_points": n_raw,
        "session_quality_flag": quality,
    }])

# COMMAND ----------

# MAGIC %md ## 3. Apply Resampling

# COMMAND ----------

curves = (
    timeseries
    .groupBy("session_id")
    .applyInPandas(resample_session, schema=output_schema)
)

# COMMAND ----------

# MAGIC %md ## 4. Write to Silver

# COMMAND ----------

(
    curves
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_session_soc_curves")
)

# COMMAND ----------

result = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_session_soc_curves")
total = result.count()
good = result.filter(F.col("session_quality_flag") == "good").count()
print(f"Total sessions: {total:,}")
print(f"Good quality:   {good:,} ({good / max(total, 1) * 100:.1f}%)")
