# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: Build Vehicle Reference Table
# MAGIC
# MAGIC Queries the customer service DB via Lakehouse Federation
# MAGIC (`fc_postgres_prod_customer.customer`) to build a denormalized
# MAGIC vehicle reference table with make, model, year, and battery capacity.
# MAGIC
# MAGIC This fills the gap in the existing dbt models which only expose
# MAGIC vehicle ID and VIN without make/model info.
# MAGIC
# MAGIC **Source tables:**
# MAGIC - `fc_postgres_prod_customer.customer.vehicle`
# MAGIC - `fc_postgres_prod_customer.customer.vehicle_make`
# MAGIC - `fc_postgres_prod_customer.customer.vehicle_model`
# MAGIC
# MAGIC **Produces:** `silver.vehicle_reference` (Delta table)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

FEDERATED_CATALOG = "fc_postgres_prod_customer"
FEDERATED_SCHEMA = "customer"
TARGET_CATALOG = "jonathan_play"
TARGET_SCHEMA = "default"

# COMMAND ----------

# MAGIC %md ## 1. Load Source Tables from Federated Catalog

# COMMAND ----------

vehicle = spark.table(f"{FEDERATED_CATALOG}.{FEDERATED_SCHEMA}.vehicle")
vehicle_make = spark.table(f"{FEDERATED_CATALOG}.{FEDERATED_SCHEMA}.vehicle_make")
vehicle_model = spark.table(f"{FEDERATED_CATALOG}.{FEDERATED_SCHEMA}.vehicle_model")

# COMMAND ----------

# MAGIC %md ## 2. Join and Denormalize

# COMMAND ----------

vehicle_ref = (
    vehicle
    .filter(F.col("deleted") == False)
    .join(vehicle_model, vehicle["vehicle_model_id"] == vehicle_model["id"], "left")
    .join(vehicle_make, vehicle_model["make_id"] == vehicle_make["id"], "left")
    .select(
        vehicle["id"].alias("vehicle_id"),
        vehicle["vin"],
        vehicle["company_id"],
        vehicle["has_active_membership"],
        vehicle_make["name"].alias("make"),
        vehicle_model["name"].alias("model"),
        vehicle_model["year"].alias("model_year"),
        vehicle_model["battery_capacity_kwh"],
    )
)

# COMMAND ----------

# MAGIC %md ## 3. Quick Validation

# COMMAND ----------

total = vehicle_ref.count()
with_make = vehicle_ref.filter(F.col("make").isNotNull()).count()
with_model = vehicle_ref.filter(F.col("model").isNotNull()).count()
with_battery = vehicle_ref.filter(F.col("battery_capacity_kwh").isNotNull()).count()

print(f"Total vehicles (non-deleted): {total:,}")
print(f"With make:                    {with_make:,} ({with_make / max(total, 1) * 100:.1f}%)")
print(f"With model:                   {with_model:,} ({with_model / max(total, 1) * 100:.1f}%)")
print(f"With battery capacity:        {with_battery:,} ({with_battery / max(total, 1) * 100:.1f}%)")

# COMMAND ----------

# Top makes by vehicle count
vehicle_ref.groupBy("make").count().orderBy(F.desc("count")).display()

# COMMAND ----------

# MAGIC %md ## 4. Write to Silver

# COMMAND ----------

(
    vehicle_ref
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_vehicle_reference")
)

# COMMAND ----------

print(f"Wrote {spark.table(f'{TARGET_CATALOG}.{TARGET_SCHEMA}.silver_vehicle_reference').count():,} rows to {TARGET_CATALOG}.{TARGET_SCHEMA}.silver_vehicle_reference")
