# greenlane-ems-pipelines

EV charging curve analysis pipeline for Greenlane's EMS platform. Produces Power-vs-SoC charge curves and piecewise linear (PWL) approximations used as constraints in the MPC LP optimizer.

## Architecture

Uses a **Medallion Architecture** on Databricks Delta tables:

```
greenlane-data-eng dbt models (existing)
  int_ems__ocpp_log_sampled_value    ← SoC, Power, Energy per meter reading
  int_ev_session                     ← session reconciliation with vehicle info
        │
        ▼
  Silver: session_timeseries         ← per-session SoC + Power time series
  Silver: session_soc_curves         ← cleaned, resampled onto uniform SoC grid
        │
        ▼
  Gold: charge_curves_by_vehicle     ← P10/P50/P90 envelopes per make/model
  Gold: charge_curve_pwl             ← piecewise linear fits → MPC LP constraints
```

## Local Development

```bash
poetry install
pytest tests/ -v
```

Business logic lives in `src/ems_pipelines/` (pure Python, no Spark dependency).
Notebooks in `notebooks/` are thin Spark orchestration — run on Databricks.

## Databricks Integration

Connect this repo via Databricks Repos (Settings > Git Integration).
Job definitions are in `jobs/`.
