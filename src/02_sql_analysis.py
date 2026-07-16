"""
02_sql_analysis.py — GridPulse Renewable Energy & EV Infrastructure Analytics
Executes complex SQL queries (Window Functions, Statistical Process Control approximations,
Multi-table joins, Spatial node aggregation) against SQLite DB and exports CSVs.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "gridpulse_energy_analytics/data/gridpulse.db"
OUTPUT_DIR = "gridpulse_energy_analytics/outputs/tables"

QUERIES = {
    "q1_grid_frequency_deviation_ewma": """
        -- Query 1: Grid Frequency Deviation Detection via Rolling Window Control Charts
        -- Identifies 15-minute intervals where grid frequency breaches the ±0.05 Hz statutory band around 50 Hz,
        -- computing rolling averages and aggregating total duration of instability by state.
        WITH frequency_windows AS (
            SELECT 
                state_id,
                timestamp_15min,
                grid_frequency_hz,
                ROUND(AVG(grid_frequency_hz) OVER (
                    PARTITION BY state_id 
                    ORDER BY timestamp_15min 
                    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                ), 4) AS rolling_3hr_avg_hz,
                ROUND(ABS(grid_frequency_hz - 50.0), 4) AS abs_deviation_hz,
                CASE 
                    WHEN ABS(grid_frequency_hz - 50.0) > 0.05 THEN 'STATUTORY VIOLATION (>0.05 Hz)'
                    WHEN ABS(grid_frequency_hz - 50.0) > 0.03 THEN 'WARNING BAND (>0.03 Hz)'
                    ELSE 'STABLE GRID (50.0 ± 0.03 Hz)'
                END AS stability_status
            FROM grid_telemetry
        )
        SELECT 
            s.state_id,
            st.name AS state_name,
            st.region,
            s.stability_status,
            COUNT(*) AS total_15min_intervals,
            ROUND(COUNT(*) * 0.25, 1) AS total_duration_hours,
            ROUND(AVG(s.abs_deviation_hz), 4) AS avg_deviation_hz,
            ROUND(MAX(s.abs_deviation_hz), 4) AS max_deviation_hz
        FROM frequency_windows s
        JOIN states st ON s.state_id = st.state_id
        WHERE s.stability_status != 'STABLE GRID (50.0 ± 0.03 Hz)'
        GROUP BY s.state_id, st.name, st.region, s.stability_status
        ORDER BY total_15min_intervals DESC;
    """,

    "q2_renewable_curtailment_weather": """
        -- Query 2: Renewable Curtailment & Weather Correlation Root-Cause Classification
        -- Joins generation output with hourly weather profiles to classify why clean power was curtailed
        -- (e.g. oversupply during peak solar/wind vs transmission bottlenecks) and estimates revenue loss.
        WITH hourly_summary AS (
            SELECT 
                g.state_id,
                st.name AS state_name,
                st.region,
                g.source_type,
                g.timestamp_hour,
                g.potential_gen_mw,
                g.actual_gen_mw,
                g.curtailed_mw,
                w.solar_irradiance_w_m2,
                w.wind_speed_ms,
                w.cloud_cover_pct,
                CASE 
                    WHEN g.potential_gen_mw > 3000 AND g.curtailed_mw / NULLIF(g.potential_gen_mw, 0) > 0.15 THEN 'GRID OVERSUPPLY (PEAK RE GENERATION)'
                    WHEN g.source_type = 'WIND' AND w.wind_speed_ms > 20 THEN 'HIGH WIND SAFETY CUTOFF'
                    WHEN g.curtailed_mw / NULLIF(g.potential_gen_mw, 0) > 0.08 THEN 'TRANSMISSION CONGESTION / BOTTLENECK'
                    ELSE 'MINIMAL / NORMAL CURTAILMENT'
                END AS curtailment_cause
            FROM plant_generation g
            JOIN weather_hourly w ON g.state_id = w.state_id AND g.timestamp_hour = w.timestamp_hour
            JOIN states st ON g.state_id = st.state_id
            WHERE g.curtailed_mw > 5.0
        )
        SELECT 
            state_id,
            state_name,
            region,
            source_type,
            curtailment_cause,
            COUNT(*) AS event_hours,
            ROUND(SUM(curtailed_mw), 1) AS total_curtailed_mwh,
            ROUND(AVG(curtailed_mw * 100.0 / NULLIF(potential_gen_mw, 0)), 2) AS avg_curtailment_pct,
            ROUND(SUM(curtailed_mw) * 0.045, 2) AS estimated_revenue_loss_lakh_rs
        FROM hourly_summary
        GROUP BY state_id, state_name, region, source_type, curtailment_cause
        ORDER BY total_curtailed_mwh DESC;
    """,

    "q3_state_re_integration_scorecard": """
        -- Query 3: State-Level Renewable Integration & Grid Stability Scorecard
        -- Ranks states based on renewable capacity share, average frequency volatility, and curtailment intensity.
        WITH state_gen AS (
            SELECT 
                state_id,
                SUM(potential_gen_mw) AS total_potential_mwh,
                SUM(curtailed_mw) AS total_curtailed_mwh,
                ROUND(SUM(curtailed_mw) * 100.0 / NULLIF(SUM(potential_gen_mw), 0), 2) AS state_curtail_pct
            FROM plant_generation
            GROUP BY state_id
        ),
        state_freq AS (
            SELECT 
                state_id,
                ROUND(MAX(grid_frequency_hz) - MIN(grid_frequency_hz), 4) AS freq_range_hz,
                ROUND(AVG(ABS(grid_frequency_hz - 50.0)), 4) AS avg_freq_deviation_hz
            FROM grid_telemetry
            GROUP BY state_id
        )
        SELECT 
            st.state_id,
            st.name AS state_name,
            st.region,
            (st.installed_solar_mw + st.installed_wind_mw) AS total_re_capacity_mw,
            st.base_demand_mw,
            ROUND((st.installed_solar_mw + st.installed_wind_mw) * 100.0 / NULLIF(st.base_demand_mw, 0), 1) AS re_penetration_ratio_pct,
            COALESCE(g.state_curtail_pct, 0.0) AS state_curtail_pct,
            COALESCE(f.freq_range_hz, 0.0) AS freq_range_hz,
            COALESCE(f.avg_freq_deviation_hz, 0.0) AS avg_freq_deviation_hz,
            DENSE_RANK() OVER (ORDER BY COALESCE(g.state_curtail_pct, 0.0) * COALESCE(f.avg_freq_deviation_hz, 0.0) DESC) AS grid_vulnerability_rank
        FROM states st
        LEFT JOIN state_gen g ON st.state_id = g.state_id
        LEFT JOIN state_freq f ON st.state_id = f.state_id
        ORDER BY grid_vulnerability_rank ASC;
    """,

    "q4_ev_charging_spatial_gap": """
        -- Query 4: Highway EV Charging Demand-Supply Spatial Gap Analysis
        -- Aggregates H3 hexagonal demand nodes and matches them against existing EV chargers
        -- to compute local supply deficit and rank priority highway corridors for new station deployment.
        WITH supply_hex AS (
            SELECT 
                hex_id,
                COUNT(*) AS existing_chargers,
                SUM(power_kw) AS total_charging_capacity_kw
            FROM ev_chargers
            GROUP BY hex_id
        ),
        hex_summary AS (
            SELECT 
                d.hex_id,
                d.highway_name,
                d.segment_km,
                d.lat,
                d.lon,
                d.daily_ev_demand,
                d.critical_demand,
                d.grid_stress_index,
                COALESCE(s.existing_chargers, 0) AS existing_chargers,
                COALESCE(s.total_charging_capacity_kw, 0) AS total_charging_capacity_kw,
                (d.daily_ev_demand - COALESCE(s.existing_chargers * 15, 0)) AS unmet_daily_trips_gap
            FROM ev_trip_demand d
            LEFT JOIN supply_hex s ON d.hex_id = s.hex_id
        )
        SELECT 
            hex_id,
            highway_name,
            segment_km,
            lat,
            lon,
            daily_ev_demand,
            critical_demand,
            existing_chargers,
            total_charging_capacity_kw,
            unmet_daily_trips_gap,
            grid_stress_index,
            ROUND(critical_demand * 1.0 / NULLIF(existing_chargers, 0), 2) AS criticality_index,
            DENSE_RANK() OVER (PARTITION BY highway_name ORDER BY (critical_demand * 1.0 / NULLIF(existing_chargers, 0)) DESC NULLS FIRST) AS highway_priority_rank
        FROM hex_summary
        WHERE unmet_daily_trips_gap > 20
        ORDER BY unmet_daily_trips_gap DESC
        LIMIT 100;
    """,

    "q5_peak_demand_generation_gap": """
        -- Query 5: Peak vs Off-Peak Generation & Demand Deficit Decomposition
        -- Analyzes 15-minute grid demand dynamics during Evening Peak (6-10 PM) vs Solar Peak (11 AM-3 PM)
        -- across regions to pinpoint where 2 GWh Battery Energy Storage Systems (BESS) yield highest value.
        WITH time_bins AS (
            SELECT 
                g.state_id,
                st.name AS state_name,
                st.region,
                g.grid_frequency_hz,
                g.total_demand_mw,
                CAST(SUBSTR(g.timestamp_15min, 12, 2) AS INTEGER) AS hour_of_day,
                CASE 
                    WHEN CAST(SUBSTR(g.timestamp_15min, 12, 2) AS INTEGER) BETWEEN 18 AND 22 THEN 'EVENING PEAK (6 PM - 10 PM)'
                    WHEN CAST(SUBSTR(g.timestamp_15min, 12, 2) AS INTEGER) BETWEEN 11 AND 15 THEN 'SOLAR PEAK (11 AM - 3 PM)'
                    ELSE 'BASE / OFF-PEAK HOURS'
                END AS diurnal_period
            FROM grid_telemetry g
            JOIN states st ON g.state_id = st.state_id
        )
        SELECT 
            state_id,
            state_name,
            region,
            diurnal_period,
            COUNT(*) AS total_15min_intervals,
            ROUND(AVG(total_demand_mw), 1) AS avg_demand_mw,
            ROUND(MAX(total_demand_mw), 1) AS max_demand_mw,
            ROUND(AVG(grid_frequency_hz), 4) AS avg_frequency_hz,
            ROUND(SUM(CASE WHEN ABS(grid_frequency_hz - 50.0) > 0.05 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS frequency_violation_rate_pct
        FROM time_bins
        GROUP BY state_id, state_name, region, diurnal_period
        ORDER BY state_name, diurnal_period;
    """
}

def run_gridpulse_sql_analysis():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to SQLite DB: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Please run 01_generate_data.py first.")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    for query_name, sql_code in QUERIES.items():
        print(f"\n--- Executing GridPulse Query: {query_name} ---")
        df = pd.read_sql_query(sql_code, conn)
        print(df.head(10).to_string(index=False))
        
        output_file = os.path.join(OUTPUT_DIR, f"{query_name}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} rows to: {output_file}")

    conn.close()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: All 5 GridPulse SQL analyses executed and exported.")

if __name__ == "__main__":
    run_gridpulse_sql_analysis()
