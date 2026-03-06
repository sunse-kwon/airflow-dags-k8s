import os
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model(ti):


    rmse = ti.xcom_pull(task_ids='model_training', key='rmse')
    run_id = ti.xcom_pull(task_ids='model_training', key='run_id')

    threshold = 0.5
    if rmse > threshold:
        logger.info(f'model RMSE {rmse} exeeds threshold {threshold}')
        raise ValueError(f'model RMSE {rmse} exeeds threshold {threshold}')
    
    MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:80"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  
    mlflow.set_experiment("automated_weather_delivery_delay_prediction_seoul")

    client = mlflow.tracking.MlflowClient()
    try:
        client.get_experiment_by_name("automated_weather_delivery_delay_prediction_seoul")
        logger.info("MLflow server accessible")
    except Exception as e:
        logger.error(f"Failed to connect to MLflow server: {e}")
        raise

    # Verify run exists
    try:
        client.get_run(run_id)
        logger.info(f"MLflow run found: {run_id}")
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Failed to find MLflow run {run_id}: {e}")
        raise

    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("validation_passed", 1)