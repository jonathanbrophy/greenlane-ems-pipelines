"""Table schema definitions for Silver and Gold Delta tables.

Column name constants to keep notebooks and utils in sync.
"""

# --- Source tables (from greenlane-data-eng dbt models) ---

SOURCE_SAMPLED_VALUE_TABLE = "int_ems__ocpp_log_sampled_value"
SOURCE_EV_SESSION_TABLE = "int_ev_session"
SOURCE_VEHICLE_TABLE = "int_customer__vehicle"

# --- Silver layer ---

SILVER_SESSION_TIMESERIES = "silver.session_timeseries"
SILVER_SESSION_SOC_CURVES = "silver.session_soc_curves"

# --- Gold layer ---

GOLD_CHARGE_CURVES = "gold.charge_curves_by_vehicle"
GOLD_CHARGE_CURVE_PWL = "gold.charge_curve_pwl"

# --- Measurand constants (OCPP) ---

MEASURAND_SOC = "SoC"
MEASURAND_POWER = "Power.Active.Import"
MEASURAND_ENERGY = "Energy.Active.Import.Register"

# --- EVSE power tier thresholds (kW) ---

EVSE_TIER_L2_MAX = 22
EVSE_TIER_DCFC_50_MAX = 100

# --- Silver: session_timeseries columns ---

class SessionTimeseries:
    SESSION_ID = "session_id"
    VEHICLE_VIN = "vehicle_vin"
    MAKE = "make"
    MODEL = "model"
    SOC = "soc"
    POWER_KW = "power_kw"
    ENERGY_WH = "energy_wh"
    EVSE_MAX_POWER_KW = "evse_max_power_kw"
    MEASUREMENT_AT_UTC = "measurement_at_utc"


# --- Silver: session_soc_curves columns ---

class SessionSocCurves:
    SESSION_ID = "session_id"
    MAKE = "make"
    MODEL = "model"
    SOC_START = "soc_start"
    SOC_END = "soc_end"
    SOC_GRID = "soc_grid"
    POWER_GRID = "power_grid"
    EVSE_MAX_POWER_KW = "evse_max_power_kw"
    N_RAW_POINTS = "n_raw_points"
    SESSION_QUALITY_FLAG = "session_quality_flag"


# --- Gold: charge_curves_by_vehicle columns ---

class ChargeCurves:
    MAKE = "make"
    MODEL = "model"
    EVSE_POWER_TIER = "evse_power_tier"
    N_SESSIONS = "n_sessions"
    P10_CURVE = "p10_curve"
    P50_CURVE = "p50_curve"
    P90_CURVE = "p90_curve"
    MAX_OBSERVED_POWER_KW = "max_observed_power_kw"
    LAST_UPDATED = "last_updated"


# --- Gold: charge_curve_pwl columns ---

class ChargeCurvePwl:
    MAKE = "make"
    MODEL = "model"
    EVSE_POWER_TIER = "evse_power_tier"
    N_SEGMENTS = "n_segments"
    PWL_BREAKPOINTS = "pwl_breakpoints"
    PWL_SLOPES = "pwl_slopes"
    FIT_RMSE = "fit_rmse"
    LAST_UPDATED = "last_updated"
