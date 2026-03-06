import sys
sys.path.insert(0, '/opt/airflow/dags/repo/dags/model_training')
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from datetime import datetime, timedelta


# add scrips directory to path
from scripts.extract import extract_data_with_columns
from scripts.preparation import prepare_data
from scripts.training import train_model
from scripts.validation import validate_model
from scripts.export import export_model
from scripts.register import push_to_model_registry


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


with DAG('model_training_seoul', default_args=default_args, start_date=datetime(2025,4,10), schedule_interval='@once', catchup=False) as dag:

    # Task to extract data with column names
    data_extraction_task = PythonOperator(
        task_id='data_extraction',
        python_callable=extract_data_with_columns,
        provide_context=True,
    )

    # Prepare data train, test split, x,y split, target log transformation for stabilize variance.
    data_preparation_task = PythonOperator(
        task_id='data_preparation',
        python_callable=prepare_data,
    )
    # 
    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=train_model,
    )

    model_validation_task = PythonOperator(
        task_id='model_validation',
        python_callable=validate_model,
    )

    export_model_task = PythonOperator(
        task_id='export_model',
        python_callable=export_model,
    )

    push_to_model_registry_task = PythonOperator(
        task_id='push_to_model_registry',
        python_callable=push_to_model_registry,
    )
    # task dependencies
    data_extraction_task >> data_preparation_task >> model_training_task >> model_validation_task >> export_model_task >> push_to_model_registry_task
