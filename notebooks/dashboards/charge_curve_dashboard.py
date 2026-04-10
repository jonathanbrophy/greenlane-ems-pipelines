# Databricks notebook source
# MAGIC %md
# MAGIC # Vehicle Charge Curve Dashboard
# MAGIC
# MAGIC Interactive visualization of Power-vs-SoC charge curves by vehicle
# MAGIC make/model. Select a vehicle from the dropdown to view its P10/P50/P90
# MAGIC quantile envelope, PWL approximation, and data coverage density.

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

TARGET_CATALOG = "jonathan_play"
TARGET_SCHEMA = "vehicle_charge_curves"

# COMMAND ----------

# MAGIC %md ## 1. Build Vehicle Selection Widgets

# COMMAND ----------

curves_df = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curves_by_vehicle")

# Get distinct make/model/tier combinations
vehicles = (
    curves_df
    .select("make", "model", "evse_power_tier")
    .distinct()
    .orderBy("make", "model", "evse_power_tier")
    .collect()
)

vehicle_labels = [
    f"{r['make']} {r['model']} ({r['evse_power_tier']})" for r in vehicles
]

dbutils.widgets.dropdown("vehicle", vehicle_labels[0], vehicle_labels, "Select Vehicle")

# COMMAND ----------

# MAGIC %md ## 2. Load Selected Vehicle Data

# COMMAND ----------

selected = dbutils.widgets.get("vehicle")

# Parse selection back to make/model/tier
for r in vehicles:
    label = f"{r['make']} {r['model']} ({r['evse_power_tier']})"
    if label == selected:
        sel_make = r["make"]
        sel_model = r["model"]
        sel_tier = r["evse_power_tier"]
        break

# Fetch the charge curve row
curve_row = (
    curves_df
    .filter(
        (F.col("make") == sel_make)
        & (F.col("model") == sel_model)
        & (F.col("evse_power_tier") == sel_tier)
    )
    .collect()[0]
)

print(f"Vehicle:    {sel_make} {sel_model}")
print(f"Power tier: {sel_tier}")
print(f"Sessions:   {curve_row['n_sessions']}")
print(f"Max power:  {curve_row['max_observed_power_kw']:.1f} kW")

# COMMAND ----------

# MAGIC %md ## 3. Load PWL Fit (if available)

# COMMAND ----------

try:
    pwl_df = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curve_pwl")
    pwl_rows = (
        pwl_df
        .filter(
            (F.col("make") == sel_make)
            & (F.col("model") == sel_model)
            & (F.col("evse_power_tier") == sel_tier)
        )
        .collect()
    )
    pwl_row = pwl_rows[0] if pwl_rows else None
except Exception:
    pwl_row = None

if pwl_row:
    print(f"PWL segments: {pwl_row['n_segments']}")
    print(f"PWL RMSE:     {pwl_row['fit_rmse']:.2f} kW")
else:
    print("No PWL fit available yet (run gold/02_fit_pwl first)")

# COMMAND ----------

# MAGIC %md ## 4. Charge Curve with P10/P50/P90 Envelope

# COMMAND ----------

# Extract curve data
p10 = curve_row["p10_curve"]
p50 = curve_row["p50_curve"]
p90 = curve_row["p90_curve"]

soc_p10 = [pt["soc"] * 100 for pt in p10]
pow_p10 = [pt["power_kw"] for pt in p10]
soc_p50 = [pt["soc"] * 100 for pt in p50]
pow_p50 = [pt["power_kw"] for pt in p50]
soc_p90 = [pt["soc"] * 100 for pt in p90]
pow_p90 = [pt["power_kw"] for pt in p90]

fig, ax = plt.subplots(figsize=(14, 6))

# P10-P90 envelope
ax.fill_between(
    soc_p10, pow_p10, pow_p90,
    alpha=0.2, color="steelblue", label="P10–P90 envelope",
)

# P50 median
ax.plot(soc_p50, pow_p50, color="steelblue", linewidth=2.5, label="P50 (median)")

# P10 and P90 bounds
ax.plot(soc_p10, pow_p10, color="steelblue", linewidth=1, linestyle=":", alpha=0.7, label="P10 / P90")
ax.plot(soc_p90, pow_p90, color="steelblue", linewidth=1, linestyle=":", alpha=0.7)

# PWL overlay
if pwl_row:
    bps = pwl_row["pwl_breakpoints"]
    soc_bp = [pt["soc"] * 100 for pt in bps]
    pow_bp = [pt["power_kw"] for pt in bps]
    ax.plot(
        soc_bp, pow_bp,
        color="red", linewidth=2, linestyle="--", marker="o",
        markersize=6, label=f"PWL fit ({pwl_row['n_segments']} segments, RMSE={pwl_row['fit_rmse']:.1f} kW)",
    )

ax.set_xlabel("State of Charge (%)", fontsize=12)
ax.set_ylabel("Power (kW)", fontsize=12)
ax.set_title(f"Charge Curve: {sel_make} {sel_model} — {sel_tier}", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## 5. Data Coverage Density
# MAGIC
# MAGIC Number of raw OCPP readings per 1% SoC bin. Higher density = more
# MAGIC confidence in the percentile estimates.

# COMMAND ----------

coverage = curve_row["coverage_count"]
soc_cov = [pt["soc"] * 100 for pt in coverage]
n_readings = [pt["n_readings"] for pt in coverage]

fig, ax = plt.subplots(figsize=(14, 3))
ax.bar(soc_cov, n_readings, width=0.8, color="steelblue", alpha=0.7)
ax.set_xlabel("State of Charge (%)", fontsize=12)
ax.set_ylabel("Raw Readings", fontsize=12)
ax.set_title(f"Data Coverage: {sel_make} {sel_model} — {sel_tier}", fontsize=14)
ax.set_xlim(0, 100)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## 6. Summary Table: All Vehicles

# COMMAND ----------

summary = (
    curves_df
    .select(
        "make",
        "model",
        "evse_power_tier",
        "n_sessions",
        "max_observed_power_kw",
        "last_updated",
    )
    .orderBy("make", "model", "evse_power_tier")
)

summary.display()
