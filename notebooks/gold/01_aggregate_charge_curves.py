# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Aggregate Charge Curves by Vehicle Make/Model
# MAGIC
# MAGIC Groups resampled session curves by (make, model, EVSE power tier) and
# MAGIC computes P10/P50/P90 quantile envelopes across the SoC grid.
# MAGIC
# MAGIC Produces: `gold.charge_curves_by_vehicle`

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

# COMMAND ----------

# MAGIC %md ## 1. Load Resampled Curves

# COMMAND ----------

curves = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_session_soc_curves")

# Filter to good-quality sessions only
good_curves = curves.filter(F.col("session_quality_flag") == "good")

# COMMAND ----------

# MAGIC %md ## 2. Add EVSE Power Tier
# MAGIC
# MAGIC Once EVSE max power is available, classify into tiers.
# MAGIC For now, use a placeholder tier.

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


def aggregate_curves(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute P10/P50/P90 quantile envelopes for a group of sessions."""
    from datetime import datetime, timezone

    # Stack all power_grid arrays into a 2D matrix
    power_matrix = np.array(pdf["power_grid"].tolist())
    # Use the SoC grid from the first session (all should be identical after resampling)
    soc_grid = np.array(pdf["soc_grid"].iloc[0])

    n_sessions = len(pdf)

    # Compute quantiles at each SoC grid point
    p10 = np.percentile(power_matrix, 10, axis=0)
    p50 = np.percentile(power_matrix, 50, axis=0)
    p90 = np.percentile(power_matrix, 90, axis=0)

    def to_curve(soc_arr, power_arr):
        return [
            {"soc": float(s), "power_kw": float(p)}
            for s, p in zip(soc_arr, power_arr)
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
        "max_observed_power_kw": float(power_matrix.max()),
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
