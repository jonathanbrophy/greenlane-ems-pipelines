# Databricks notebook source
# MAGIC %md
# MAGIC # Vehicle Charge Curve Dashboard
# MAGIC
# MAGIC Interactive visualization of Power-vs-SoC charge curves by vehicle
# MAGIC make/model. Use the dropdown in the chart to switch between vehicles
# MAGIC instantly.

# COMMAND ----------

# MAGIC %md ## 1. Load All Data

# COMMAND ----------

from pyspark.sql import functions as F

TARGET_CATALOG = "jonathan_play"
TARGET_SCHEMA = "vehicle_charge_curves"

curves_df = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curves_by_vehicle")
all_curves = curves_df.collect()

if not all_curves:
    raise RuntimeError(
        "No data in gold_charge_curves_by_vehicle. "
        "Run the pipeline first: silver/02 -> gold/01 -> (optional) gold/02"
    )

try:
    pwl_df = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curve_pwl")
    all_pwl = pwl_df.collect()
except Exception:
    all_pwl = []

# Build lookup dicts keyed by (make, model, tier)
curve_lookup = {}
for r in all_curves:
    key = (r["make"], r["model"], r["evse_power_tier"])
    curve_lookup[key] = r

pwl_lookup = {}
for r in all_pwl:
    key = (r["make"], r["model"], r["evse_power_tier"])
    pwl_lookup[key] = r

vehicle_keys = sorted(curve_lookup.keys(), key=lambda k: (k[0] or "", k[1] or "", k[2] or ""))

print(f"Loaded {len(all_curves)} vehicle/tier groups, {len(all_pwl)} PWL fits")

# COMMAND ----------

# MAGIC %md ## 2. Charge Curve Dashboard
# MAGIC
# MAGIC Use the dropdown at the top-left of the chart to switch vehicles.
# MAGIC Hover for values, drag to zoom, double-click to reset.

# COMMAND ----------

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Each vehicle gets a set of traces. We'll show/hide them via the dropdown.
# Traces per vehicle: envelope_upper, envelope_lower, p10, p50, p90, [pwl], coverage_bar
# We use a consistent number of traces per vehicle (7) so indexing is simple.
# If no PWL, the PWL trace is empty.

TRACES_PER_VEHICLE = 7  # envelope_upper, envelope_lower, p10, p50, p90, pwl, coverage

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.75, 0.25],
)

all_visibility = []  # list of visibility arrays, one per vehicle

for idx, key in enumerate(vehicle_keys):
    curve_row = curve_lookup[key]
    pwl_row = pwl_lookup.get(key)
    make, model, tier = key
    visible = idx == 0  # only first vehicle visible initially

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

    # Envelope upper bound
    fig.add_trace(
        go.Scatter(
            x=soc_p90, y=pow_p90,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
            visible=visible,
        ),
        row=1, col=1,
    )
    # Envelope lower bound (filled to upper)
    fig.add_trace(
        go.Scatter(
            x=soc_p10, y=pow_p10,
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(70, 130, 180, 0.2)",
            name="P10-P90 envelope",
            hoverinfo="skip",
            visible=visible,
            legendgroup="envelope",
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
            visible=visible,
        ),
        row=1, col=1,
    )
    # P50 median
    fig.add_trace(
        go.Scatter(
            x=soc_p50, y=pow_p50,
            mode="lines",
            line=dict(color="steelblue", width=3),
            name="P50 (median)",
            hovertemplate="SoC: %{x:.0f}%<br>P50: %{y:.1f} kW<extra></extra>",
            visible=visible,
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
            visible=visible,
        ),
        row=1, col=1,
    )
    # PWL overlay (empty if not available)
    if pwl_row:
        bps = pwl_row["pwl_breakpoints"]
        soc_bp = [pt["soc"] * 100 for pt in bps]
        pow_bp = [pt["power_kw"] for pt in bps]
        pwl_name = f"PWL ({pwl_row['n_segments']} seg, RMSE={pwl_row['fit_rmse']:.1f} kW)"
    else:
        soc_bp, pow_bp = [], []
        pwl_name = "PWL (not fitted)"

    fig.add_trace(
        go.Scatter(
            x=soc_bp, y=pow_bp,
            mode="lines+markers",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=8, color="red"),
            name=pwl_name,
            hovertemplate="SoC: %{x:.1f}%<br>PWL: %{y:.1f} kW<extra></extra>",
            visible=visible if soc_bp else False,
        ),
        row=1, col=1,
    )
    # Coverage bar chart
    fig.add_trace(
        go.Bar(
            x=soc_cov, y=n_readings,
            marker_color="steelblue",
            opacity=0.7,
            name="Raw readings",
            hovertemplate="SoC: %{x:.0f}%<br>Readings: %{y:,}<extra></extra>",
            visible=visible,
        ),
        row=2, col=1,
    )

# Build dropdown buttons
n_vehicles = len(vehicle_keys)
total_traces = n_vehicles * TRACES_PER_VEHICLE
buttons = []

for idx, key in enumerate(vehicle_keys):
    make, model, tier = key
    curve_row = curve_lookup[key]
    pwl_row = pwl_lookup.get(key)

    # Visibility: only this vehicle's traces are visible
    vis = [False] * total_traces
    start = idx * TRACES_PER_VEHICLE
    for j in range(TRACES_PER_VEHICLE):
        # PWL trace (index 5) only visible if it has data
        if j == 5 and not pwl_lookup.get(key):
            vis[start + j] = False
        else:
            vis[start + j] = True

    title = (
        f"Charge Curve: {make} {model} -- {tier}  |  "
        f"{curve_row['n_sessions']} sessions  |  "
        f"Max {curve_row['max_observed_power_kw']:.0f} kW"
    )

    buttons.append(dict(
        label=f"{make} {model} ({tier})",
        method="update",
        args=[
            {"visible": vis},
            {"annotations": [
                dict(
                    text=title,
                    xref="paper", yref="paper",
                    x=0.5, y=1.0, xanchor="center",
                    showarrow=False, font=dict(size=14),
                ),
                dict(
                    text="Data Coverage (raw OCPP readings per 1% SoC bin)",
                    xref="paper", yref="paper",
                    x=0.5, y=0.22, xanchor="center",
                    showarrow=False, font=dict(size=14),
                ),
            ]},
        ],
    ))

# Set initial title
first_key = vehicle_keys[0]
first_row = curve_lookup[first_key]
initial_title = (
    f"Charge Curve: {first_key[0]} {first_key[1]} -- {first_key[2]}  |  "
    f"{first_row['n_sessions']} sessions  |  "
    f"Max {first_row['max_observed_power_kw']:.0f} kW"
)

fig.update_layout(
    height=750,
    template="plotly_white",
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.06,
        xanchor="right", x=1,
    ),
    margin=dict(t=120, l=60, r=20, b=40),
    annotations=[
        dict(
            text=initial_title,
            xref="paper", yref="paper",
            x=0.5, y=1.0, xanchor="center",
            showarrow=False, font=dict(size=14),
        ),
        dict(
            text="Data Coverage (raw OCPP readings per 1% SoC bin)",
            xref="paper", yref="paper",
            x=0.5, y=0.22, xanchor="center",
            showarrow=False, font=dict(size=14),
        ),
    ],
    updatemenus=[dict(
        type="dropdown",
        direction="down",
        showactive=True,
        x=0.0,
        xanchor="left",
        y=1.18,
        yanchor="top",
        buttons=buttons,
        bgcolor="white",
        bordercolor="#ccc",
        font=dict(size=12),
    )],
)

fig.update_xaxes(title_text="State of Charge (%)", row=2, col=1, range=[0, 100], dtick=10)
fig.update_xaxes(range=[0, 100], dtick=10, row=1, col=1)
fig.update_yaxes(title_text="Power (kW)", row=1, col=1, rangemode="tozero")
fig.update_yaxes(title_text="Readings", row=2, col=1, rangemode="tozero")

fig.show()

# COMMAND ----------

# MAGIC %md ## 3. Summary Table: All Vehicles

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
