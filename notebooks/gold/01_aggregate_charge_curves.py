# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Aggregate Charge Curves by Vehicle Make/Model
# MAGIC
# MAGIC Groups cleaned session curves by (make, model, EVSE power tier),
# MAGIC interpolates each session onto a universal SoC grid (0–1 in 1% steps),
# MAGIC and computes P10/P50/P90 quantile envelopes.
# MAGIC
# MAGIC Produces: `gold_charge_curves_by_vehicle`

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

TARGET_CATALOG = "jonathan_play"
TARGET_SCHEMA = "vehicle_charge_curves"

# Universal SoC grid: 0% to 100% in 1% steps
SOC_GRID = np.linspace(0.0, 1.0, 101)

# COMMAND ----------

# MAGIC %md ## 1. Load Cleaned Curves

# COMMAND ----------

curves = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_session_soc_curves")

# Filter to good-quality sessions only
good_curves = curves.filter(F.col("session_quality_flag") == "good")

# COMMAND ----------

# MAGIC %md ## 2. Add EVSE Power Tier

# COMMAND ----------

with_tier = good_curves.withColumn(
    "evse_power_tier",
    F.when(F.col("evse_max_power_kw").isNull(), "unknown")
     .when(F.col("evse_max_power_kw") <= 22, "L2")
     .when(F.col("evse_max_power_kw") <= 100, "DCFC_50")
     .otherwise("DCFC_150+"),
)

# COMMAND ----------

# MAGIC %md ## 3. Compute Quantile Envelopes per Group
# MAGIC
# MAGIC Each session's raw SoC/Power samples are interpolated onto the universal
# MAGIC grid (0–1 in 1% steps). Only the portion of the grid covered by a session's
# MAGIC SoC range is filled; out-of-range points are NaN. Percentiles are computed
# MAGIC with `np.nanpercentile` so sessions with partial coverage still contribute.

# COMMAND ----------

output_schema = StructType([
    StructField("make", StringType(), True),
    StructField("model", StringType(), True),
    StructField("evse_power_tier", StringType(), False),
    StructField("n_sessions", IntegerType(), False),
    StructField("p10_curve", ArrayType(
        StructType([
            StructField("soc", FloatType()),
            StructField("power_kw", FloatType()),
        ])
    ), False),
    StructField("p50_curve", ArrayType(
        StructType([
            StructField("soc", FloatType()),
            StructField("power_kw", FloatType()),
        ])
    ), False),
    StructField("p90_curve", ArrayType(
        StructType([
            StructField("soc", FloatType()),
            StructField("power_kw", FloatType()),
        ])
    ), False),
    StructField("max_observed_power_kw", FloatType(), False),
    StructField("last_updated", TimestampType(), False),
])


# Broadcast the universal grid so workers can access it
SOC_GRID_LIST = SOC_GRID.tolist()


def aggregate_curves(pdf: pd.DataFrame) -> pd.DataFrame:
    """Interpolate sessions onto universal grid, then compute P10/P50/P90."""
    from datetime import datetime, timezone

    soc_grid = np.array(SOC_GRID_LIST)
    n_sessions = len(pdf)

    # Interpolate each session onto the universal grid
    power_matrix = np.full((n_sessions, len(soc_grid)), np.nan)
    for i, (_, row) in enumerate(pdf.iterrows()):
        soc_raw = np.array(row["soc_values"])
        power_raw = np.array(row["power_values"])

        # Only interpolate within the session's SoC range
        mask = (soc_grid >= soc_raw.min()) & (soc_grid <= soc_raw.max())
        power_matrix[i, mask] = np.interp(soc_grid[mask], soc_raw, power_raw)

    # Compute quantiles, ignoring NaN (sessions that don't cover the full range)
    p10 = np.nanpercentile(power_matrix, 10, axis=0)
    p50 = np.nanpercentile(power_matrix, 50, axis=0)
    p90 = np.nanpercentile(power_matrix, 90, axis=0)

    def to_curve(soc_arr, power_arr):
        return [
            {"soc": float(s), "power_kw": float(p)}
            for s, p in zip(soc_arr, power_arr)
            if not np.isnan(p)
        ]

    row = pdf.iloc[0]
    return pd.DataFrame([{
        "make": row.get("make"),
        "model": row.get("model"),
        "evse_power_tier": row["evse_power_tier"],
        "n_sessions": n_sessions,
        "p10_curve": to_curve(soc_grid, p10),
        "p50_curve": to_curve(soc_grid, p50),
        "p90_curve": to_curve(soc_grid, p90),
        "max_observed_power_kw": float(np.nanmax(power_matrix)),
        "last_updated": datetime.now(timezone.utc),
    }])

# COMMAND ----------

aggregated = (
    with_tier
    .groupBy("make", "model", "evse_power_tier")
    .applyInPandas(aggregate_curves, schema=output_schema)
)

# COMMAND ----------

# MAGIC %md ## 4. Write to Gold

# COMMAND ----------

(
    aggregated
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curves_by_vehicle")
)

# COMMAND ----------

result = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curves_by_vehicle")
result.select("make", "model", "evse_power_tier", "n_sessions", "max_observed_power_kw").display()
