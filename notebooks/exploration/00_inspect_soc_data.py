# Databricks notebook source
# MAGIC %md
# MAGIC # Inspect SoC Data Availability
# MAGIC
# MAGIC **Run this first** before building any pipeline logic.
# MAGIC
# MAGIC Validates:
# MAGIC 1. What % of sessions have SoC measurand in OCPP MeterValues
# MAGIC 2. SoC value ranges and quality (0-100 vs 0-1, nulls, anomalies)
# MAGIC 3. Power measurand co-occurrence with SoC
# MAGIC 4. Session-to-vehicle join coverage
# MAGIC 5. Sample raw curves for visual inspection

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC
# MAGIC Set the catalog and schema where dbt models are materialized.

# COMMAND ----------

CATALOG = "prod"
SCHEMA = "public"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md ## 1. SoC Measurand Availability

# COMMAND ----------

sampled_values = spark.table("int_ems__ocpp_log_sampled_value")

total_sessions = sampled_values.select("driivz__ev_transaction_id").distinct().count()

soc_sessions = (
    sampled_values
    .filter(F.col("measurand") == "SoC")
    .select("driivz__ev_transaction_id")
    .distinct()
    .count()
)

print(f"Total sessions with meter values: {total_sessions:,}")
print(f"Sessions with SoC measurand:      {soc_sessions:,}")
print(f"SoC coverage:                     {soc_sessions / total_sessions * 100:.1f}%")

# COMMAND ----------

# MAGIC %md ## 2. SoC Value Ranges and Quality

# COMMAND ----------

soc_values = sampled_values.filter(F.col("measurand") == "SoC")

soc_val = F.col("value").cast("float")

soc_stats = soc_values.agg(
    F.min(soc_val).alias("min_soc"),
    F.max(soc_val).alias("max_soc"),
    F.avg(soc_val).alias("avg_soc"),
    F.stddev(soc_val).alias("stddev_soc"),
    F.count(F.when(soc_val.isNull(), 1)).alias("null_count"),
    F.count(F.when(soc_val < 0, 1)).alias("negative_count"),
    F.count(F.when(soc_val > 100, 1)).alias("over_100_count"),
    F.count("*").alias("total_readings"),
)

soc_stats.display()

# COMMAND ----------

# Check the unit field for SoC readings (cast from VARIANT to STRING)
soc_values.groupBy(F.col("unit").cast("string")).count().orderBy(F.desc("count")).display()

# COMMAND ----------

# Distribution of SoC values (histogram buckets)
soc_values.select(
    F.floor(F.col("value").cast("float") / 10).alias("soc_decile")
).groupBy("soc_decile").count().orderBy("soc_decile").display()

# COMMAND ----------

# MAGIC %md ## 3. Power Measurand Co-occurrence with SoC

# COMMAND ----------

# For sessions that have SoC, do they also have Power.Active.Import?
soc_session_ids = (
    sampled_values
    .filter(F.col("measurand") == "SoC")
    .select("driivz__ev_transaction_id")
    .distinct()
)

power_in_soc_sessions = (
    sampled_values
    .join(soc_session_ids, on="driivz__ev_transaction_id")
    .filter(F.col("measurand") == "Power.Active.Import")
    .select("driivz__ev_transaction_id")
    .distinct()
    .count()
)

print(f"Sessions with SoC:              {soc_sessions:,}")
print(f"Of those, also have Power:      {power_in_soc_sessions:,}")
print(f"Co-occurrence rate:             {power_in_soc_sessions / max(soc_sessions, 1) * 100:.1f}%")

# COMMAND ----------

# Check what measurands are available per meter_value reading
# (are SoC and Power reported in the same MeterValues call?)
soc_logs = (
    sampled_values
    .filter(F.col("measurand") == "SoC")
    .select("ocpp_log_sid", "meter_value_idx")
    .distinct()
)

co_measurands = (
    sampled_values
    .join(soc_logs, on=["ocpp_log_sid", "meter_value_idx"])
    .select(F.col("measurand").cast("string").alias("measurand_str"))
    .groupBy("measurand_str")
    .count()
    .orderBy(F.desc("count"))
)

co_measurands.display()

# COMMAND ----------

# MAGIC %md ## 4. Session-to-Vehicle Join Coverage

# COMMAND ----------

ev_sessions = spark.table("int_ev_session")

sessions_with_vin = ev_sessions.filter(F.col("vehicle_vin").isNotNull()).count()
total_ev_sessions = ev_sessions.count()

print(f"Total EV sessions:              {total_ev_sessions:,}")
print(f"Sessions with VIN:              {sessions_with_vin:,}")
print(f"VIN coverage:                   {sessions_with_vin / max(total_ev_sessions, 1) * 100:.1f}%")

# COMMAND ----------

# Cross-reference: sessions that have BOTH SoC data AND a VIN
soc_and_vin = (
    ev_sessions
    .filter(F.col("vehicle_vin").isNotNull())
    .join(soc_session_ids, ev_sessions["driivz__ev_transaction_id"] == soc_session_ids["driivz__ev_transaction_id"])
    .count()
)

print(f"Sessions with SoC AND VIN:      {soc_and_vin:,}")

# COMMAND ----------

# MAGIC %md ## 5. Sample Raw Curves

# COMMAND ----------

# Pick a session that has both SoC and Power readings
sample_session_id = (
    sampled_values
    .filter(F.col("measurand") == "SoC")
    .filter(F.col("driivz__ev_transaction_id").isNotNull())
    .select("driivz__ev_transaction_id")
    .limit(1)
    .collect()[0][0]
)

print(f"Sample session: {sample_session_id}")

# COMMAND ----------

# Get all sampled values for this session, ordered by time
sample_data = (
    sampled_values
    .filter(F.col("driivz__ev_transaction_id") == sample_session_id)
    .filter(F.col("measurand").isin("SoC", "Power.Active.Import", "Energy.Active.Import.Register"))
    .select(
        "meter_value_at_utc",
        "measurand",
        "value",
        "unit",
        "context",
        "meter_value_idx",
        "sampled_value_idx",
    )
    .orderBy("meter_value_at_utc", "measurand")
)

sample_data.display()

# COMMAND ----------

# Pivot to wide format for plotting: one row per timestamp with SoC, Power, Energy columns
# Pre-select and cast to avoid VARIANT type issues with pivot/groupBy
sample_narrow = (
    sampled_values
    .filter(F.col("driivz__ev_transaction_id") == sample_session_id)
    .filter(F.col("context").cast("string").isin("Sample.Periodic") | F.col("context").isNull())
    .select(
        F.col("ocpp_log_sid"),
        F.col("meter_value_idx"),
        F.col("meter_value_at_utc"),
        F.col("measurand").cast("string").alias("measurand"),
        F.col("value").cast("float").alias("value"),
    )
)

sample_wide = (
    sample_narrow
    .groupBy("ocpp_log_sid", "meter_value_idx", "meter_value_at_utc")
    .pivot("measurand", ["SoC", "Power.Active.Import", "Energy.Active.Import.Register"])
    .agg(F.first("value"))
    .orderBy("meter_value_at_utc")
    .withColumnRenamed("SoC", "soc")
    .withColumnRenamed("Power.Active.Import", "power_w")
    .withColumnRenamed("Energy.Active.Import.Register", "energy_wh")
)

sample_wide.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick Plot: Power vs SoC for Sample Session
# MAGIC
# MAGIC Convert to pandas for a quick matplotlib plot.

# COMMAND ----------

import matplotlib.pyplot as plt

pdf = sample_wide.filter(F.col("soc").isNotNull() & F.col("power_w").isNotNull()).toPandas()

if not pdf.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(pdf["soc"], pdf["power_w"] / 1000, s=10, alpha=0.7)
    ax.set_xlabel("SoC (%)")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Power vs SoC — Session {sample_session_id}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No rows with both SoC and Power — check data availability above.")
