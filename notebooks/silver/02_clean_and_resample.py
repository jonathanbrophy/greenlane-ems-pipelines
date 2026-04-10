# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: Clean and Filter Session Curves
# MAGIC
# MAGIC Reads `silver_session_timeseries`, applies quality filters, and produces
# MAGIC one row per session with raw SoC/Power arrays and a quality flag.
# MAGIC
# MAGIC Produces: `silver_session_soc_curves` — one row per session with
# MAGIC `soc_values` and `power_values` arrays (raw samples, no resampling).

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

# MAGIC %md ## 2. Define Quality Filter and Pandas UDF
# MAGIC
# MAGIC Functions are defined inline so they're available on worker nodes
# MAGIC when executed via `applyInPandas`.

# COMMAND ----------

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
    StructField("soc_values", ArrayType(FloatType()), False),
    StructField("power_values", ArrayType(FloatType()), False),
    StructField("n_raw_points", IntegerType(), False),
    StructField("session_quality_flag", StringType(), False),
])


def clean_session(pdf: pd.DataFrame) -> pd.DataFrame:
    """Clean a single session: sort, drop nulls, classify quality, store raw arrays."""
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

    # Drop sessions with fewer than 3 points
    if n_raw < 3:
        return pd.DataFrame()

    row = pdf.iloc[0]
    return pd.DataFrame([{
        "session_id": row["session_id"],
        "make": row.get("make"),
        "model": row.get("model"),
        "evse_max_power_kw": row.get("evse_max_power_kw"),
        "soc_start": float(soc.min()),
        "soc_end": float(soc.max()),
        "soc_values": soc.tolist(),
        "power_values": power.tolist(),
        "n_raw_points": n_raw,
        "session_quality_flag": quality,
    }])

# COMMAND ----------

# MAGIC %md ## 3. Apply Cleaning

# COMMAND ----------

curves = (
    timeseries
    .groupBy("session_id")
    .applyInPandas(clean_session, schema=output_schema)
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
