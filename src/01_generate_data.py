
import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import h3

def generate_gridpulse_database(db_path: str = "gridpulse_energy_analytics/data/gridpulse.db", seed: int = 108):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting GridPulse synthetic data generation...")
    rng = np.random.default_rng(seed)

    states_data = [
        (1, "Rajasthan", "North", 18500, 5200, 14500, 27.0, 74.0),
        (2, "Gujarat", "West", 11200, 10800, 19000, 22.3, 71.5),
        (3, "Tamil Nadu", "South", 7800, 10400, 17500, 11.1, 78.7),
        (4, "Karnataka", "South", 8900, 5600, 15800, 15.3, 75.7),
        (5, "Maharashtra", "West", 5400, 5100, 26000, 19.8, 75.8),
        (6, "Andhra Pradesh", "South", 4800, 4200, 12200, 15.9, 79.7),
        (7, "Madhya Pradesh", "West", 3500, 2800, 11500, 23.5, 78.5),
        (8, "Telangana", "South", 4900, 200, 13800, 18.1, 79.0),
        (9, "Uttar Pradesh", "North", 2800, 50, 25000, 26.8, 80.9),
        (10, "Punjab", "North", 1200, 0, 13500, 31.1, 75.3),
        (11, "Haryana", "North", 1100, 0, 12000, 29.0, 76.1),
        (12, "West Bengal", "East", 250, 50, 10500, 23.8, 87.9),
        (13, "Odisha", "East", 480, 50, 6800, 20.9, 85.1),
        (14, "Bihar", "East", 210, 0, 7200, 25.4, 85.3),
        (15, "Chhattisgarh", "West", 420, 0, 5900, 21.3, 81.9),
        (16, "Kerala", "South", 450, 80, 4800, 10.8, 76.3),
        (17, "Jharkhand", "East", 120, 0, 3500, 23.6, 85.5),
        (18, "Assam", "Northeast", 150, 0, 2400, 26.2, 92.9),
        (19, "Himachal Pradesh", "North", 110, 0, 2100, 31.9, 77.2),
        (20, "Uttarakhand", "North", 320, 0, 2600, 30.1, 79.0),
        (21, "Delhi", "North", 280, 0, 7800, 28.6, 77.2),
        (22, "Goa", "West", 40, 0, 700, 15.3, 74.1),
        (23, "Jammu & Kashmir", "North", 60, 0, 2200, 33.8, 76.6),
        (24, "Tripura", "Northeast", 25, 0, 350, 23.8, 91.3),
        (25, "Meghalaya", "Northeast", 10, 0, 400, 25.5, 91.9),
        (26, "Manipur", "Northeast", 15, 0, 300, 24.8, 93.9),
        (27, "Nagaland", "Northeast", 5, 0, 200, 26.1, 94.6),
        (28, "Arunachal Pradesh", "Northeast", 12, 0, 180, 28.2, 94.7)
    ]
    states_df = pd.DataFrame(states_data, columns=[
        "state_id", "name", "region", "installed_solar_mw", "installed_wind_mw",
        "base_demand_mw", "lat", "lon"
    ])


    plants_list = []
    plant_id_counter = 1
    for _, st in states_df.iterrows():

        n_solar = max(1, int(st["installed_solar_mw"] / 250))
        for p in range(n_solar):
            cap = round(st["installed_solar_mw"] / n_solar + rng.normal(0, 50), 1)
            plants_list.append({
                "plant_id": plant_id_counter,
                "state_id": st["state_id"],
                "name": f"{st['name'][:4].upper()}_SOLAR_PARK_{p+1:02d}",
                "source_type": "SOLAR",
                "capacity_mw": max(10.0, cap),
                "lat": round(st["lat"] + rng.uniform(-1.2, 1.2), 4),
                "lon": round(st["lon"] + rng.uniform(-1.2, 1.2), 4),
                "panel_efficiency": round(rng.uniform(0.18, 0.22), 4)
            })
            plant_id_counter += 1
            

        if st["installed_wind_mw"] > 0:
            n_wind = max(1, int(st["installed_wind_mw"] / 200))
            for p in range(n_wind):
                cap = round(st["installed_wind_mw"] / n_wind + rng.normal(0, 40), 1)
                plants_list.append({
                    "plant_id": plant_id_counter,
                    "state_id": st["state_id"],
                    "name": f"{st['name'][:4].upper()}_WIND_FARM_{p+1:02d}",
                    "source_type": "WIND",
                    "capacity_mw": max(10.0, cap),
                    "lat": round(st["lat"] + rng.uniform(-1.0, 1.0), 4),
                    "lon": round(st["lon"] + rng.uniform(-1.0, 1.0), 4),
                    "panel_efficiency": 0.0
                })
                plant_id_counter += 1
    plants_df = pd.DataFrame(plants_list)


    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating ~490k hourly weather & solar/wind profile records...")
    start_date = pd.to_datetime("2024-01-01")
    n_days = 730
    hours_range = pd.date_range(start=start_date, periods=n_days * 24, freq="h")
    

    n_weather = len(states_df) * len(hours_range)
    state_ids_rep = np.repeat(states_df["state_id"].values, len(hours_range))
    timestamps_rep = np.tile(hours_range.strftime("%Y-%m-%d %H:%M:%S"), len(states_df))
    hours_of_day = np.tile(hours_range.hour, len(states_df))
    months_of_year = np.tile(hours_range.month, len(states_df))
    

    solar_base = np.sin(np.pi * (hours_of_day - 6) / 12.0)
    solar_base = np.where((hours_of_day >= 6) & (hours_of_day <= 18), solar_base, 0.0)
    

    monsoon_flag = np.where(np.isin(months_of_year, [6, 7, 8, 9]), 1.0, 0.2)
    cloud_cover = np.clip(rng.beta(2, 5, size=n_weather) * 100.0 * monsoon_flag + rng.normal(0, 10, size=n_weather), 0, 100).round(1)
    

    irradiance = (solar_base * 950.0 * (1.0 - cloud_cover / 130.0) + rng.normal(0, 15, size=n_weather)).round(1)
    irradiance = np.clip(irradiance, 0, 1100)
    

    wind_speed = rng.weibull(2.1, size=n_weather) * 6.5 + (monsoon_flag * 3.5)
    wind_speed = np.clip(wind_speed, 0, 28).round(2)

    weather_df = pd.DataFrame({
        "state_id": state_ids_rep,
        "timestamp_hour": timestamps_rep,
        "solar_irradiance_w_m2": irradiance,
        "wind_speed_ms": wind_speed,
        "cloud_cover_pct": cloud_cover,
        "temperature_c": (26.0 + 8.0 * solar_base - (cloud_cover / 10.0) + rng.normal(0, 2, size=n_weather)).round(1)
    })


    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating ~980k hourly renewable curtailment & generation panel...")
    state_solar_cap = states_df.set_index("state_id")["installed_solar_mw"].to_dict()
    state_wind_cap = states_df.set_index("state_id")["installed_wind_mw"].to_dict()
    state_demand_base = states_df.set_index("state_id")["base_demand_mw"].to_dict()
    

    solar_df = weather_df[["state_id", "timestamp_hour", "solar_irradiance_w_m2", "cloud_cover_pct"]].copy()
    solar_df["source_type"] = "SOLAR"
    solar_df["installed_mw"] = solar_df["state_id"].map(state_solar_cap)
    solar_df["potential_gen_mw"] = (solar_df["installed_mw"] * (solar_df["solar_irradiance_w_m2"] / 1000.0) * 0.85).round(1)
    
    wind_df = weather_df[weather_df["state_id"].map(state_wind_cap) > 0][["state_id", "timestamp_hour", "wind_speed_ms"]].copy()
    wind_df["source_type"] = "WIND"
    wind_df["installed_mw"] = wind_df["state_id"].map(state_wind_cap)

    wind_df["potential_gen_mw"] = (wind_df["installed_mw"] * np.clip((wind_df["wind_speed_ms"] - 3.0) / 10.0, 0, 1.0) * 0.90).round(1)
    
    gen_df = pd.concat([
        solar_df[["state_id", "timestamp_hour", "source_type", "potential_gen_mw"]],
        wind_df[["state_id", "timestamp_hour", "source_type", "potential_gen_mw"]]
    ], ignore_index=True)
    

    is_rj_tn = gen_df["state_id"].isin([1, 3])
    is_peak_gen = gen_df["potential_gen_mw"] > 3000.0
    curtail_factor = np.where(is_rj_tn & is_peak_gen, rng.uniform(0.18, 0.38, size=len(gen_df)), rng.uniform(0.01, 0.06, size=len(gen_df)))
    
    gen_df["curtailed_mw"] = (gen_df["potential_gen_mw"] * curtail_factor).round(1)
    gen_df["actual_gen_mw"] = (gen_df["potential_gen_mw"] - gen_df["curtailed_mw"]).round(1)


    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating ~1.96M 15-minute grid SCADA telemetry rows...")

    sampled_states = states_df[states_df["state_id"].isin([1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 16, 21])].copy()
    timestamps_15m = pd.date_range(start=start_date, periods=n_days * 96, freq="15min")
    
    n_scada = len(sampled_states) * len(timestamps_15m)
    scada_state_ids = np.repeat(sampled_states["state_id"].values, len(timestamps_15m))
    scada_times = np.tile(timestamps_15m.strftime("%Y-%m-%d %H:%M:%S"), len(sampled_states))
    scada_hours = np.tile(timestamps_15m.hour, len(sampled_states))
    

    demand_multiplier = 1.0 + 0.25 * np.sin(np.pi * (scada_hours - 6) / 12.0) + np.where((scada_hours >= 18) & (scada_hours <= 22), 0.3, 0.0)
    scada_base_demand = scada_state_ids
    scada_demand_map = sampled_states.set_index("state_id")["base_demand_mw"].to_dict()
    demand_mw = (np.vectorize(scada_demand_map.get)(scada_state_ids) * demand_multiplier + rng.normal(0, 200, size=n_scada)).round(1)
    

    gen_df["hour_key"] = gen_df["timestamp_hour"].str.slice(0, 13)
    state_hour_curtail = gen_df.groupby(["state_id", "hour_key"])["curtailed_mw"].sum().to_dict()


    scada_hour_keys = [t[:13] for t in scada_times]
    mapped_curtail = np.array([state_hour_curtail.get((sid, hk), 0.0) for sid, hk in zip(scada_state_ids, scada_hour_keys)])
    

    is_high_re_state = np.isin(scada_state_ids, [1, 2, 3, 4])
    ramp_instability = np.where(is_high_re_state & np.isin(scada_hours, [12, 13, 14, 17, 18]), rng.normal(0, 0.045, size=n_scada), rng.normal(0, 0.015, size=n_scada))
    grid_freq = (50.0 - (mapped_curtail / 18000.0) + ramp_instability).round(4)

    grid_telemetry_df = pd.DataFrame({
        "state_id": scada_state_ids,
        "timestamp_15min": scada_times,
        "grid_frequency_hz": grid_freq,
        "total_demand_mw": demand_mw
    })


    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating EV highway charging demand & H3 hexagonal spatial nodes...")
    highways = [
        ("NH-44 (North-South Corridor: Delhi-Agra-Nagpur-Bangalore-Kanyakumari)", 3745, 28.6, 77.2, 8.1, 77.5),
        ("NH-48 (Golden Quadrilateral West: Delhi-Jaipur-Ahmedabad-Mumbai-Bangalore)", 2807, 28.6, 77.2, 12.9, 77.5),
        ("NH-19 (Delhi-Kolkata via Varanasi)", 1435, 28.6, 77.2, 22.5, 88.3),
        ("NH-16 (East Coast: Kolkata-Bhubaneswar-Visakhapatnam-Chennai)", 1711, 22.5, 88.3, 13.0, 80.2),
        ("NH-27 (East-West Corridor: Porbandar-Jhansi-Muzaffarpur-Guwahati)", 3507, 21.6, 69.6, 26.1, 91.7),
        ("NH-65 (Pune-Hyderabad-Machilipatnam)", 841, 18.5, 73.8, 16.2, 81.1),
        ("NH-52 (Sangrur-Indore-Ankleshwar)", 1440, 30.2, 75.8, 21.6, 73.0),
        ("NH-66 (Panvel-Goa-Mangalore-Kanyakumari Coastal West)", 1622, 18.9, 73.1, 8.1, 77.5)
    ]
    
    ev_demand_nodes = []
    chargers_list = []
    node_id_counter = 1
    charger_id_counter = 1
    
    for hw_name, dist_km, lats, lons, late, lone in highways:
        n_nodes = int(dist_km / 15) + 1  # ~1 node every 15 km
        for n in range(n_nodes):
            frac = n / n_nodes
            lat = lats + (late - lats) * frac + rng.normal(0, 0.05)
            lon = lons + (lone - lons) * frac + rng.normal(0, 0.05)

            hex_id = h3.latlng_to_cell(lat, lon, 7)
            

            is_metro_vicinity = 1 if (frac < 0.15 or frac > 0.85 or abs(frac - 0.5) < 0.08) else 0
            daily_trips = int(rng.uniform(180, 650) if is_metro_vicinity else rng.uniform(30, 140))
            critical_demand = int(daily_trips * rng.uniform(0.25, 0.45) if not is_metro_vicinity else daily_trips * 0.12)
            grid_stress = round(rng.uniform(0.65, 0.95) if not is_metro_vicinity else rng.uniform(0.25, 0.60), 2)
            
            ev_demand_nodes.append({
                "node_id": node_id_counter,
                "hex_id": hex_id,
                "highway_name": hw_name,
                "segment_km": round(n * 15.0, 1),
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "daily_ev_demand": daily_trips,
                "critical_demand": critical_demand,
                "grid_stress_index": grid_stress
            })
            

            if is_metro_vicinity and rng.random() > 0.3:
                chargers_list.append({
                    "charger_id": charger_id_counter,
                    "node_id": node_id_counter,
                    "hex_id": hex_id,
                    "highway_name": hw_name,
                    "lat": round(lat + rng.normal(0, 0.01), 4),
                    "lon": round(lon + rng.normal(0, 0.01), 4),
                    "power_kw": int(rng.choice([22, 60, 120, 150], p=[0.4, 0.35, 0.15, 0.10])),
                    "operator": rng.choice(["Tata Power EZ", "Statiq", "Jio-bp Pulse", "Zeon Charging", "ChargeZone"])
                })
                charger_id_counter += 1
            node_id_counter += 1

    ev_demand_df = pd.DataFrame(ev_demand_nodes)
    chargers_df = pd.DataFrame(chargers_list)


    print(f"[{datetime.now().strftime('%H:%M:%S')}] Writing tables to SQLite database at {db_path}...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    states_df.to_sql("states", conn, index=False)
    plants_df.to_sql("power_plants", conn, index=False)
    weather_df.to_sql("weather_hourly", conn, index=False)
    gen_df.to_sql("plant_generation", conn, index=False)
    grid_telemetry_df.to_sql("grid_telemetry", conn, index=False)
    ev_demand_df.to_sql("ev_trip_demand", conn, index=False)
    chargers_df.to_sql("ev_chargers", conn, index=False)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Indexing GridPulse tables...")
    conn.execute("CREATE INDEX idx_gt_state_time ON grid_telemetry(state_id, timestamp_15min);")
    conn.execute("CREATE INDEX idx_pg_state_source ON plant_generation(state_id, source_type);")
    conn.execute("CREATE INDEX idx_ev_hex ON ev_trip_demand(hex_id);")
    conn.commit()
    conn.close()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: GridPulse synthetic database generated (~3.5M total records).")

if __name__ == "__main__":
    generate_gridpulse_database()
