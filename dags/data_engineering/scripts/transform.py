from datetime import datetime
from airflow.providers.postgres.hooks.postgres import PostgresHook
import json
import logging


def process_weather_data(ti):
    # Fetch data from XCom
    weather_data = ti.xcom_pull(task_ids='fetch_weather_data')
    # weather_data is a list of tuples: [(raw_json, base_date, base_time, nx, ny), ...]
    
    # Prepare data for bulk insertion
    processed_data = []
    for item in weather_data:
        raw_json, base_date, base_time, nx, ny = item
        # Convert date and time to strings, escape quotes in raw_json
        processed_row = (
            str(raw_json).replace("'", "''"),  # Escape single quotes for SQL
            base_date.strftime('%Y-%m-%d'),    # Convert date to string
            base_time,    
            nx,                                # Integer
            ny                                 # Integer
        )
        processed_data.append(processed_row)
    
    return processed_data  # List of tuples for bulk insertion


category_mapping = {
                    "T1H": ("Temperature", "Celsius"),
                    "RN1":("1HourRainfall", "mm"),
                    "UUU":("WindSpeedEastWest", "m/s"),
                    "VVV":("WindSpeedSouthNorth", "m/s"),
                    "REH": ("Humidity", "%"),
                    "PTY": ("RainType", "0no1rain2rainSnow3snow5rainDrop6blowRainSnow7blowSnow"),
                    "VEC": ("WindDirection", "deg"),
                    "WSD": ("WindSpeed", "m/s"),
                    }
    
# Hardcoded holidays for 2025 (South Korea, approximate for lunar holidays)
holiday_dates = {
    "2025-01-01": "New Year's Day",
    "2025-01-28": "Seollal (Lunar New Year)",  # Approximate, adjust manually
    "2025-01-29": "Seollal Holiday",          # Approximate, adjust manually
    "2025-03-01": "Independence Movement Day",
    "2025-05-05": "Children's Day",
    "2025-06-06": "Memorial Day",
    "2025-08-15": "National Liberation Day",
    "2025-10-03": "National Foundation Day",
    "2025-10-05": "Chuseok Holiday D1",          
    "2025-10-06": "Chuseok Holiday D2",          
    "2025-10-07": "Chuseok Holiday D3",         
    "2025-10-08": "Chuseok Holiday D4",          
    "2025-10-09": "Hangeul Day",
    "2025-12-25": "Christmas Day",
}

location_context = {
    "Seoul": ("1171064600", "Seoul", "Songpa-gu, Jangji-dong", 63,125),
    "Incheon":("2811052000", "Incheon", "Jung-gu, Yeonan-dong", 53,124),
    "Icheon":("4150034000", "Icheon", "Majang-myeon", 66,120),
    "Daegu":("2771038000", "Daegu", "Dalsung-gun, Guji-myeon", 86,86)
}

# Helper functions
def transform_weather_data(ti):
    """Transform raw_json into dimension and fact data with holiday and location logic."""
    pg_hook = PostgresHook(postgres_conn_id='weather_connection')
    staging_data = ti.xcom_pull(task_ids='extract_from_staging')
    
    logging.info(f"Staging data: {staging_data}")

    if not staging_data:
        return {"processed_ids": []}
    
    transformed_data = {
        "dim_date": [],
        "dim_time": [],
        "dim_location": [],
        "dim_category": [],
        "fact_weather": [],
        "processed_ids": []
    }
    
    for row in staging_data:
        staging_id, raw_json, base_date, base_time, nx, ny = row

        logging.info(f"Processing row: staging_id={staging_id}, raw_json={raw_json}, base_date={base_date}")

        # Parse JSON
        # weather_data = json.loads(raw_json)
        weather_data = raw_json  # Remove json.loads(raw_json)
        
        # Dim Date with hardcoded holiday logic
        date_obj = base_date  # No strptime needed
        base_date_str = base_date.strftime('%Y-%m-%d')
        is_holiday = "TRUE" if base_date_str in holiday_dates else "FALSE"
        transformed_data["dim_date"].append({
            "base_date": base_date_str,
            "year": date_obj.year,
            "month": date_obj.month,
            "day": date_obj.day,
            "day_of_week": date_obj.strftime('%A'),
            "is_holiday": is_holiday
        })
        
        # Dim Time
        hour = int(base_time[:2])
        transformed_data["dim_time"].append({
            "base_time": base_time,
            "hour": hour
        })
        
        # Dim Location with lookup from location_context
        location_match = None
        for city, (admin_district_code, city_name, sub_address, loc_nx, loc_ny) in location_context.items():
            if nx == loc_nx and ny == loc_ny:
                location_match = (admin_district_code, city_name, sub_address)
                break
        
        if location_match:
            admin_district_code, city, sub_address = location_match
        else:
            admin_district_code, city, sub_address = "UNKNOWN", "UNKNOWN", "UNKNOWN"
        
        transformed_data["dim_location"].append({
            "nx": nx,
            "ny": ny,
            "admin_district_code": admin_district_code,
            "city": city,
            "sub_address": sub_address,
            "effective_date": base_date_str,
            "expiration_date": None,
            "is_current": True
        })
        
        # Dim Category and Fact Weather using category_mapping
        category = weather_data.get("category")
        obsr_value = weather_data.get("obsrValue")
        

        if category in category_mapping:
            logging.info(f"check category_mapping: {category}")

            description, unit = category_mapping[category]
            logging.info(f"description,unit: {description}, {unit}")

            transformed_data["dim_category"].append({
                "category_code": category,
                "category_description": description,
                "unit": unit,
                "effective_date": base_date_str,
                "expiration_date": None,
                "is_current": True
            })

            transformed_data["fact_weather"].append({
                "base_date": base_date,
                "base_time": base_time,
                "nx": nx,
                "ny": ny,
                "category_code": category,
                "measurement_value": float(obsr_value)
            })
        
        transformed_data["processed_ids"].append(staging_id)
    
    ti.xcom_push(key='transformed_data', value=transformed_data)
    return transformed_data