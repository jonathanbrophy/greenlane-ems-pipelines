# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Fit Piecewise Linear Approximations
# MAGIC
# MAGIC Reads P50 charge curves from `gold.charge_curves_by_vehicle` and fits
# MAGIC piecewise linear (PWL) approximations using `pwlf`.
# MAGIC
# MAGIC The resulting breakpoints and slopes map directly to MPC LP constraints:
# MAGIC - `pwl_breakpoints`: SoC thresholds with power limits -> P(SoC) <= f(SoC)
# MAGIC - `pwl_slopes`: dP/dSoC rates for each segment
# MAGIC
# MAGIC Produces: `gold.charge_curve_pwl`

# COMMAND ----------

# MAGIC %pip install pwlf

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import pwlf
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

# Number of PWL segments — tunable by controls team
# Fewer = simpler LP constraints, more = better fidelity
N_SEGMENTS = 5

# COMMAND ----------

# MAGIC %md ## 1. Load Aggregated Curves

# COMMAND ----------

curves = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curves_by_vehicle")

# COMMAND ----------

# MAGIC %md ## 2. Define PWL Fitting Functions and Pandas UDF
# MAGIC
# MAGIC Functions are defined inline so they're available on worker nodes
# MAGIC when executed via `applyInPandas`.

# COMMAND ----------

def fit_pwl(soc_grid, power_values, n_segments=5):
    """Fit a piecewise linear model to a charge curve."""
    model = pwlf.PiecewiseLinFit(soc_grid, power_values)
    model.fit(n_segments)
    return model


def pwl_to_breakpoints(model):
    """Extract breakpoints as (soc, power_kw) pairs."""
    soc_breaks = model.fit_breaks
    power_breaks = model.predict(soc_breaks)
    return [(float(s), float(p)) for s, p in zip(soc_breaks, power_breaks)]


def pwl_to_slopes(model):
    """Extract slopes (dP/dSoC) for each segment."""
    return [float(s) for s in model.calc_slopes()]


def compute_fit_rmse(model, soc_grid, power_values):
    """Compute RMSE between the PWL fit and the original curve."""
    predicted = model.predict(soc_grid)
    return float(np.sqrt(np.mean((predicted - power_values) ** 2)))

# COMMAND ----------

output_schema = StructType([
    StructField("make", StringType(), True),
    StructField("model", StringType(), True),
    StructField("evse_power_tier", StringType(), False),
    StructField("n_segments", IntegerType(), False),
    StructField("pwl_breakpoints", ArrayType(
        StructType([
            StructField("soc", FloatType()),
            StructField("power_kw", FloatType()),
        ])
    ), False),
    StructField("pwl_slopes", ArrayType(FloatType()), False),
    StructField("fit_rmse", FloatType(), False),
    StructField("last_updated", TimestampType(), False),
])


def fit_pwl_for_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """Fit a PWL approximation to the P50 curve for a single make/model group."""
    from datetime import datetime, timezone

    row = pdf.iloc[0]
    p50_curve = row["p50_curve"]

    soc_grid = np.array([pt["soc"] for pt in p50_curve])
    power_values = np.array([pt["power_kw"] for pt in p50_curve])

    try:
        model = fit_pwl(soc_grid, power_values, n_segments=N_SEGMENTS)
        breakpoints = pwl_to_breakpoints(model)
        slopes = pwl_to_slopes(model)
        rmse = compute_fit_rmse(model, soc_grid, power_values)
    except Exception as e:
        # If fitting fails (e.g., not enough data), return with NaN RMSE
        print(f"PWL fit failed for {row.get('make')}/{row.get('model')}: {e}")
        breakpoints = [
            (float(soc_grid[0]), float(power_values[0])),
            (float(soc_grid[-1]), float(power_values[-1])),
        ]
        slopes = [0.0]
        rmse = float("nan")

    bp_structs = [{"soc": s, "power_kw": p} for s, p in breakpoints]

    return pd.DataFrame([{
        "make": row.get("make"),
        "model": row.get("model"),
        "evse_power_tier": row["evse_power_tier"],
        "n_segments": N_SEGMENTS,
        "pwl_breakpoints": bp_structs,
        "pwl_slopes": slopes,
        "fit_rmse": rmse,
        "last_updated": datetime.now(timezone.utc),
    }])

# COMMAND ----------

pwl_results = (
    curves
    .groupBy("make", "model", "evse_power_tier")
    .applyInPandas(fit_pwl_for_group, schema=output_schema)
)

# COMMAND ----------

# MAGIC %md ## 3. Write to Gold

# COMMAND ----------

(
    pwl_results
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curve_pwl")
)

# COMMAND ----------

# MAGIC %md ## 4. Inspect Results

# COMMAND ----------

result = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curve_pwl")
result.select("make", "model", "evse_power_tier", "n_segments", "fit_rmse").display()

# COMMAND ----------

# MAGIC %md ### Visualize PWL Fit vs P50 Curve

# COMMAND ----------

import matplotlib.pyplot as plt

# Pick the first row for visualization
sample = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curves_by_vehicle").limit(1).collect()
pwl_sample = result.limit(1).collect()

if sample and pwl_sample:
    p50 = sample[0]["p50_curve"]
    p10 = sample[0]["p10_curve"]
    p90 = sample[0]["p90_curve"]
    bps = pwl_sample[0]["pwl_breakpoints"]

    soc_p50 = [pt["soc"] for pt in p50]
    pow_p50 = [pt["power_kw"] for pt in p50]
    soc_p10 = [pt["soc"] for pt in p10]
    pow_p10 = [pt["power_kw"] for pt in p10]
    soc_p90 = [pt["soc"] for pt in p90]
    pow_p90 = [pt["power_kw"] for pt in p90]
    soc_bp = [pt["soc"] for pt in bps]
    pow_bp = [pt["power_kw"] for pt in bps]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(soc_p10, pow_p10, pow_p90, alpha=0.2, color="steelblue", label="P10-P90 envelope")
    ax.plot(soc_p50, pow_p50, color="steelblue", linewidth=2, label="P50 median")
    ax.plot(soc_bp, pow_bp, color="red", linewidth=2, linestyle="--", marker="o", label="PWL approximation")
    ax.set_xlabel("State of Charge")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Charge Curve: {sample[0]['make']} {sample[0]['model']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No data to visualize yet.")
