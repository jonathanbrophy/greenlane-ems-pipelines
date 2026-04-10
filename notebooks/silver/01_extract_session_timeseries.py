# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: Extract Session Timeseries
# MAGIC
# MAGIC Reads from dbt intermediate tables produced by `greenlane-data-eng`:
# MAGIC - `int_ems__ocpp_log_sampled_value` — periodic meter readings (SoC, Power, Energy)
# MAGIC - `int_ev_session` — session reconciliation with vehicle info
# MAGIC
# MAGIC Produces: `silver.session_timeseries` — one row per (session, measurement) with
# MAGIC SoC, power, and energy pivoted into columns.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

SOURCE_CATALOG = "prod"
SOURCE_SCHEMA = "public"
TARGET_CATALOG = "jonathan_play"
TARGET_SCHEMA = "vehicle_charge_curves"

# COMMAND ----------

# MAGIC %md ## 1. Load Source Tables

# COMMAND ----------

sampled_values = spark.table(f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.int_ems__ocpp_log_sampled_value")
ev_sessions = spark.table(f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.ev_session")

# COMMAND ----------

# MAGIC %md ## 2. Pivot Measurands to Wide Format
# MAGIC
# MAGIC The sampled_value table has one row per measurand per meter reading.
# MAGIC We need SoC, Power, and Energy on the same row, joined by
# MAGIC `ocpp_log_sid` + `meter_value_idx`.

# COMMAND ----------

# Filter to periodic sample readings (exclude Transaction.Begin/End)
# Pre-select and cast VARIANT columns to native types before pivot
periodic = (
    sampled_values
    .filter(
        (F.col("call_action") == "MeterValues")
        & (F.col("context").cast("string").isin("Sample.Periodic") | F.col("context").isNull())
    )
    .select(
        F.col("driivz__ev_transaction_id"),
        F.col("driivz__charger_id"),
        F.col("driivz__charger_connector_idx"),
        F.col("ocpp_log_sid"),
        F.col("meter_value_idx"),
        F.col("meter_value_at_utc"),
        F.col("measurand").cast("string").alias("measurand"),
        F.col("value").cast("float").alias("value"),
    )
)

# Pivot: one row per (ocpp_log_sid, meter_value_idx) with SoC, Power, Energy as columns
wide = (
    periodic
    .groupBy(
        "driivz__ev_transaction_id",
        "driivz__charger_id",
        "driivz__charger_connector_idx",
        "ocpp_log_sid",
        "meter_value_idx",
        "meter_value_at_utc",
    )
    .pivot(
        "measurand",
        ["SoC", "Power.Active.Import", "Energy.Active.Import.Register"],
    )
    .agg(F.first("value"))
    .withColumnRenamed("SoC", "soc_raw")
    .withColumnRenamed("Power.Active.Import", "power_w")
    .withColumnRenamed("Energy.Active.Import.Register", "energy_wh")
)

# COMMAND ----------

# MAGIC %md ## 3. Filter to Sessions with SoC Data

# COMMAND ----------

# Keep only rows where SoC was reported
with_soc = wide.filter(F.col("soc_raw").isNotNull())

# COMMAND ----------

# MAGIC %md ## 4. Join to EV Session and Vehicle Reference

# COMMAND ----------

session_info = ev_sessions.select(
    F.col("driivz__ev_transaction_id"),
    F.col("customer__vehicle_id"),
    F.col("vehicle_vin"),
)

# Vehicle reference table built by 00_build_vehicle_reference.py
vehicle_ref = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_vehicle_reference").select(
    F.col("vehicle_id"),
    F.col("make"),
    F.col("model"),
    F.col("model_year"),
    F.col("battery_capacity_kwh"),
)

timeseries = (
    with_soc
    .join(session_info, on="driivz__ev_transaction_id", how="inner")
    .join(
        vehicle_ref,
        F.col("customer__vehicle_id") == vehicle_ref["vehicle_id"],
        how="left",
    )
)

# COMMAND ----------

# MAGIC %md ## 5. Normalize and Write

# COMMAND ----------

result = timeseries.select(
    F.col("driivz__ev_transaction_id").alias("session_id"),
    F.col("vehicle_vin"),
    F.col("make"),
    F.col("model"),
    F.col("model_year"),
    F.col("battery_capacity_kwh"),
    # Normalize SoC: if reported as 0-100, convert to 0-1
    F.when(F.col("soc_raw") > 1.0, F.col("soc_raw") / 100.0)
     .otherwise(F.col("soc_raw"))
     .alias("soc"),
    # Convert power from W to kW
    (F.col("power_w") / 1000.0).alias("power_kw"),
    F.col("energy_wh"),
    # EVSE max power — will need connector spec lookup in future
    F.lit(None).cast("float").alias("evse_max_power_kw"),
    F.col("meter_value_at_utc").alias("measurement_at_utc"),
)

# COMMAND ----------

# Write to Silver Delta table
(
    result
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_session_timeseries")
)

# COMMAND ----------

print(f"Wrote {spark.table(f'{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_session_timeseries').count():,} rows to {TARGET_CATALOG}.{TARGET_SCHEMA}.silver_session_timeseries")
