# Databricks notebook source
# MAGIC %md
# MAGIC # Vehicle Charge Curve Dashboard
# MAGIC
# MAGIC Interactive visualization of Power-vs-SoC charge curves by vehicle
# MAGIC make/model. Select a vehicle from the dropdown — the charts update
# MAGIC instantly without re-running the notebook.

# COMMAND ----------

# MAGIC %md ## 1. Load All Data

# COMMAND ----------

from pyspark.sql import functions as F

TARGET_CATALOG = "jonathan_play"
TARGET_SCHEMA = "vehicle_charge_curves"

# Load everything into memory once
curves_df = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.gold_charge_curves_by_vehicle")
all_curves = curves_df.collect()

if not all_curves:
    raise RuntimeError(
        "No data in gold_charge_curves_by_vehicle. "
        "Run the pipeline first: silver/02 -> gold/01 -> (optional) gold/02"
    )

# Try loading PWL fits
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

# Build sorted labels
vehicle_keys = sorted(curve_lookup.keys(), key=lambda k: (k[0] or "", k[1] or "", k[2] or ""))
vehicle_labels = {
    f"{k[0]} {k[1]} ({k[2]})": k for k in vehicle_keys
}

print(f"Loaded {len(all_curves)} vehicle/tier groups, {len(all_pwl)} PWL fits")

# COMMAND ----------

# MAGIC %md ## 2. Interactive Dashboard
# MAGIC
# MAGIC Select a vehicle from the dropdown to update the charts instantly.

# COMMAND ----------

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots


def build_figure(key):
    """Build the Plotly figure for a given (make, model, tier) key."""
    curve_row = curve_lookup[key]
    pwl_row = pwl_lookup.get(key)
    make, model, tier = key

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

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=[
            f"Charge Curve: {make} {model} -- {tier}  |  "
            f"{curve_row['n_sessions']} sessions  |  "
            f"Max {curve_row['max_observed_power_kw']:.0f} kW",
            "Data Coverage (raw OCPP readings per 1% SoC bin)",
        ],
    )

    # P10-P90 envelope
    fig.add_trace(
        go.Scatter(
            x=soc_p90, y=pow_p90,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=soc_p10, y=pow_p10,
            mode="lines", line=dict(width=0),
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

    # P50 median
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

    # Coverage bar chart
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

    fig.update_layout(
        height=700,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(t=80),
    )
    fig.update_xaxes(title_text="State of Charge (%)", row=2, col=1, range=[0, 100], dtick=10)
    fig.update_xaxes(range=[0, 100], dtick=10, row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Readings", row=2, col=1, rangemode="tozero")

    return fig


# --- Build interactive widget ---

dropdown = widgets.Dropdown(
    options=list(vehicle_labels.keys()),
    value=list(vehicle_labels.keys())[0],
    description="Vehicle:",
    style={"description_width": "60px"},
    layout=widgets.Layout(width="400px"),
)

fig_widget = go.FigureWidget(build_figure(vehicle_labels[dropdown.value]))


def on_vehicle_change(change):
    key = vehicle_labels[change["new"]]
    new_fig = build_figure(key)

    with fig_widget.batch_update():
        # Update traces
        for i, trace in enumerate(new_fig.data):
            if i < len(fig_widget.data):
                fig_widget.data[i].x = trace.x
                fig_widget.data[i].y = trace.y
                if hasattr(trace, "name"):
                    fig_widget.data[i].name = trace.name

        # Update titles
        fig_widget.layout.annotations[0].text = new_fig.layout.annotations[0].text
        fig_widget.layout.annotations[1].text = new_fig.layout.annotations[1].text


dropdown.observe(on_vehicle_change, names="value")

display(widgets.VBox([dropdown, fig_widget]))

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
