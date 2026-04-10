# Databricks notebook source
# MAGIC %md
# MAGIC # Vehicle Charge Curve Dashboard
# MAGIC
# MAGIC Interactive visualization of Power-vs-SoC charge curves by vehicle
# MAGIC make/model. Select a vehicle from the dropdown to view its P10/P50/P90
# MAGIC quantile envelope, PWL approximation, and data coverage density.

# COMMAND ----------

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

if not vehicle_labels:
    raise RuntimeError(
        "No data in gold_charge_curves_by_vehicle. "
        "Run the pipeline first: silver/02 -> gold/01 -> (optional) gold/02"
    )

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

# MAGIC %md ## 4. Charge Curve with P10/P50/P90 Envelope and Coverage

# COMMAND ----------

# Extract curve data
p10 = curve_row["p10_curve"]
p50 = curve_row["p50_curve"]
p90 = curve_row["p90_curve"]
coverage = curve_row["coverage_count"]

soc_p10 = [pt["soc"] * 100 for pt in p10]
pow_p10 = [pt["power_kw"] for pt in p10]
soc_p50 = [pt["soc"] * 100 for pt in p50]
pow_p50 = [pt["power_kw"] for pt in p50]
soc_p90 = [pt["soc"] * 100 for pt in p90]
pow_p90 = [pt["power_kw"] for pt in p90]
soc_cov = [pt["soc"] * 100 for pt in coverage]
n_readings = [pt["n_readings"] for pt in coverage]

# Build combined figure: charge curve on top, coverage bar chart below
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.75, 0.25],
    subplot_titles=[
        f"Charge Curve: {sel_make} {sel_model} -- {sel_tier}",
        "Data Coverage (raw OCPP readings per 1% SoC bin)",
    ],
)

# --- Top chart: Charge curve ---

# P10-P90 envelope (filled area)
fig.add_trace(
    go.Scatter(
        x=soc_p90, y=pow_p90,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(
        x=soc_p10, y=pow_p10,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(70, 130, 180, 0.2)",
        name="P10-P90 envelope",
        hoverinfo="skip",
    ),
    row=1, col=1,
)

# P10 line
fig.add_trace(
    go.Scatter(
        x=soc_p10, y=pow_p10,
        mode="lines",
        line=dict(color="steelblue", width=1, dash="dot"),
        name="P10",
        hovertemplate="SoC: %{x:.0f}%<br>P10: %{y:.1f} kW<extra></extra>",
    ),
    row=1, col=1,
)

# P50 median line
fig.add_trace(
    go.Scatter(
        x=soc_p50, y=pow_p50,
        mode="lines",
        line=dict(color="steelblue", width=3),
        name="P50 (median)",
        hovertemplate="SoC: %{x:.0f}%<br>P50: %{y:.1f} kW<extra></extra>",
    ),
    row=1, col=1,
)

# P90 line
fig.add_trace(
    go.Scatter(
        x=soc_p90, y=pow_p90,
        mode="lines",
        line=dict(color="steelblue", width=1, dash="dot"),
        name="P90",
        hovertemplate="SoC: %{x:.0f}%<br>P90: %{y:.1f} kW<extra></extra>",
    ),
    row=1, col=1,
)

# PWL overlay
if pwl_row:
    bps = pwl_row["pwl_breakpoints"]
    soc_bp = [pt["soc"] * 100 for pt in bps]
    pow_bp = [pt["power_kw"] for pt in bps]
    fig.add_trace(
        go.Scatter(
            x=soc_bp, y=pow_bp,
            mode="lines+markers",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=8, color="red"),
            name=f"PWL ({pwl_row['n_segments']} seg, RMSE={pwl_row['fit_rmse']:.1f} kW)",
            hovertemplate="SoC: %{x:.1f}%<br>PWL: %{y:.1f} kW<extra></extra>",
        ),
        row=1, col=1,
    )

# --- Bottom chart: Coverage bar chart ---

fig.add_trace(
    go.Bar(
        x=soc_cov, y=n_readings,
        marker_color="steelblue",
        opacity=0.7,
        name="Raw readings",
        hovertemplate="SoC: %{x:.0f}%<br>Readings: %{y:,}<extra></extra>",
    ),
    row=2, col=1,
)

# Layout
fig.update_layout(
    height=700,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=80),
)

fig.update_xaxes(
    title_text="State of Charge (%)", row=2, col=1,
    range=[0, 100], dtick=10,
)
fig.update_xaxes(range=[0, 100], dtick=10, row=1, col=1)
fig.update_yaxes(title_text="Power (kW)", row=1, col=1, rangemode="tozero")
fig.update_yaxes(title_text="Readings", row=2, col=1, rangemode="tozero")

fig.show()

# COMMAND ----------

# MAGIC %md ## 5. Summary Table: All Vehicles

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
