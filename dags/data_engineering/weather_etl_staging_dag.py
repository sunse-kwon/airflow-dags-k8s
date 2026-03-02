import sys
sys.path.insert(0, '/opt/airflow/dags/repo/dags/data_engineering')
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# add scrips directory to path
from scripts.extract import fetch_weather_data
from scripts.transform import process_weather_data
from scripts.load import stage_weather_data


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
with DAG('weather_etl_staging_dag', default_args=default_args, start_date=datetime(2025,3,15), schedule_interval='10 * * * *', catchup=False) as dag:
    ## fetch data from api using python operator
    fetch_task = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data,
    )
    
    # Process the data and prepare it for SQL
    process_task = PythonOperator(
        task_id='process_weather_data',
        python_callable=process_weather_data,
        provide_context=True,  # Allows access to ti (TaskInstance)
    )

    stage_task = PythonOperator(
        task_id='stage_weather_data',
        python_callable=stage_weather_data,
        provide_context=True
    )

    # # define Task dependencies
    fetch_task >> process_task >> stage_task