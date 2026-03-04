import sys
sys.path.insert(0, '/opt/airflow/dags/repo/dags/data_engineering')
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta


# add scrips directory to path
from scripts.transform import transform_weather_data
from scripts.load import load_to_master_tables
from scripts.utils import mark_processed


# define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email':['sunse523@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 3,
    'timezone': 'KST',
    'retry_delay': timedelta(minutes=5)
}

# define Dag
with DAG('weather_etl_main_dag', default_args=default_args, start_date=datetime(2025,3,15), schedule_interval='10 * * * *', catchup=False) as dag:
    
    # Wait for upstream DAG's final task to complete
    wait_for_upstream = ExternalTaskSensor(
        task_id='wait_for_upstream_dag',
        external_dag_id='weather_etl_staging_dag',
        external_task_id='stage_weather_data',   
        # execution_delta=timedelta(minutes=5),  # Adjust based on schedules
        timeout=60*60,
        mode='reschedule'
    )
    # Extract data
    extract_staging_task = SQLExecuteQueryOperator(
        task_id='extract_from_staging',
        conn_id='weather_connection',
        sql="""
            SELECT staging_id, raw_json, base_date, base_time, nx, ny
            FROM staging_weather
            WHERE processed = FALSE
        """,
        do_xcom_push=True,
    )

    # Transform data
    transform_task = PythonOperator(
        task_id='transform_weather_data',
        python_callable=transform_weather_data,
    )
    
    # Load to master tables
    load_task = PythonOperator(
        task_id='load_to_master_tables',
        python_callable=load_to_master_tables,
    )
    
    # Mark as processed 
    mark_processed_task = PythonOperator(
        task_id='mark_processed',
        python_callable=mark_processed,
    )
    
    # Define task dependencies
    wait_for_upstream >> extract_staging_task >> transform_task >> load_task >> mark_processed_task