from airflow.providers.postgres.hooks.postgres import PostgresHook


def mark_processed(ti):
    """Mark processed rows in staging_weather using transformed_data['processed_ids']."""
    pg_hook = PostgresHook(postgres_conn_id='weather_connection')
    transformed_data = ti.xcom_pull(task_ids='transform_weather_data', key='transformed_data')
    processed_ids = transformed_data.get('processed_ids', [])
    
    if processed_ids:
        # Use a parameterized query with array input
        query = """
            UPDATE staging_weather
            SET processed = TRUE
            WHERE staging_id = ANY(%s)
        """
        pg_hook.run(query, parameters=(processed_ids,))