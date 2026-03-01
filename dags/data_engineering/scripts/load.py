from airflow.hooks.postgres_hook import PostgresHook
import logging


def stage_weather_data(ti):
    """Load weather data into PostgreSQL using PostgresHook."""
    # Pull the processed data from XCom
    weather_data = ti.xcom_pull(task_ids='process_weather_data')
    if not weather_data:
        raise ValueError("No weather data received from process_task")

    # Initialize PostgresHook with your connection ID
    pg_hook = PostgresHook(postgres_conn_id='weather_connection')
    
    pg_hook.insert_rows(
        table="staging_weather",
        rows=weather_data,
        target_fields=["raw_json", "base_date", "base_time", "nx", "ny"],
        replace=False
    )


def load_to_master_tables(ti):

    """Load transformed data into dimension and fact tables efficiently."""
    pg_hook = PostgresHook(postgres_conn_id='weather_connection')
    transformed_data = ti.xcom_pull(task_ids='transform_weather_data', key='transformed_data')
    
    logging.info(f"Transformed data received: {transformed_data}")

    if not transformed_data:
        return
    
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # --- Batch Insert Dimensions ---
    # Dim Date
    date_values = [
        (d["base_date"], d["year"], d["month"], d["day"], d["day_of_week"], d["is_holiday"])
        for d in transformed_data["dim_date"]
        ]
    if date_values:
        cursor.executemany("""
            INSERT INTO dim_date (base_date, year, month, day, day_of_week, is_holiday)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, date_values)
    
    # Dim Time
    time_values = [(t["base_time"], t["hour"]) for t in transformed_data["dim_time"]]
    if time_values:
        cursor.executemany("""
            INSERT INTO dim_time (base_time, hour)
            VALUES (%s, %s)
        """, time_values)
    
    # Dim Location
    location_values = [
        (l["nx"], l["ny"], l["admin_district_code"], l["city"], l["sub_address"], 
         l["effective_date"], l["expiration_date"], l["is_current"])
        for l in transformed_data["dim_location"]
        ]
    if location_values:
        cursor.executemany("""
            INSERT INTO dim_location (nx, ny, admin_district_code, city, sub_address, effective_date, expiration_date, is_current)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, location_values)
    
    # Dim Category
    category_values = [
        (c["category_code"], c["category_description"], c["unit"], c["effective_date"], 
         c["expiration_date"], c["is_current"])
        for c in transformed_data["dim_category"]
        ]
    if category_values:
        cursor.executemany("""
            INSERT INTO dim_category (category_code, category_description, unit, effective_date, expiration_date, is_current)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, category_values)
    
    # --- Pre-Fetch Dimension IDs ---
    # Fetch all date_ids
    fact_weather = transformed_data["fact_weather"]
    logging.info(f"Fact weather: {fact_weather}")
    base_dates = list(set(f["base_date"] for f in transformed_data["fact_weather"]))
    logging.info(f"Base dates: {base_dates}")
    cursor.execute("""
        SELECT base_date, date_id 
        FROM dim_date 
        WHERE base_date IN %s
    """, (tuple(base_dates),))
    date_id_map = dict(cursor.fetchall())
    
    # Fetch all time_ids
    base_times = list(set(f["base_time"] for f in transformed_data["fact_weather"]))
    logging.info(f"Base times: {base_times}")
    cursor.execute("""
        SELECT base_time, time_id 
        FROM dim_time 
        WHERE base_time IN %s
    """, (tuple(base_times),))
    time_id_map = dict(cursor.fetchall())
    
    # Fetch all location_ids
    nx_ny_pairs = list(set((f["nx"], f["ny"]) for f in transformed_data["fact_weather"]))
    logging.info(f"Nx ny pairs: {nx_ny_pairs}")
    cursor.execute("""
        SELECT nx, ny, location_id 
        FROM dim_location 
        WHERE (nx, ny) IN %s AND is_current = TRUE
    """, (tuple(nx_ny_pairs),))
    location_id_map = {(row[0], row[1]): row[2] for row in cursor.fetchall()}
    
    # Fetch all category_ids
    category_codes = list(set(f["category_code"] for f in transformed_data["fact_weather"]))
    logging.info(f"Category codes: {category_codes}")
    cursor.execute("""
        SELECT category_code, category_id 
        FROM dim_category 
        WHERE category_code IN %s AND is_current = TRUE
    """, (tuple(category_codes),))
    category_id_map = dict(cursor.fetchall())
    
    # --- Batch Insert Fact Weather ---
    fact_values = []
    for fact in transformed_data["fact_weather"]:
        date_id = date_id_map.get(fact["base_date"])
        time_id = time_id_map.get(fact["base_time"])
        location_id = location_id_map.get((fact["nx"], fact["ny"]))
        category_id = category_id_map.get(fact["category_code"])
        
        if all([date_id, time_id, location_id, category_id]):  # Ensure no NULLs
            fact_values.append((date_id, time_id, location_id, category_id, fact["measurement_value"]))
    # test smpt , change name of fact table  fact_weather_measurement
    if fact_values:
        cursor.executemany("""
            INSERT INTO fact_weather_measurement (date_id, time_id, location_id, category_id, measurement_value)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, fact_values)
    
    # Commit the transaction
    conn.commit()
    cursor.close()