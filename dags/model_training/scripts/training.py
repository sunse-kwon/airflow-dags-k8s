import io
import os
import json
import random
import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import logging

random.seed(42)         
np.random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BUCKET = 'feature-store-785685275217'


def read_from_s3(s3_key: str) -> pd.DataFrame:
    s3 = boto3.client('s3', region_name='ap-northeast-2')
    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buffer)
    buffer.seek(0)
    return pd.read_parquet(buffer)


# def save_json_to_s3(data: dict, s3_key: str):
#     s3 = boto3.client('s3', region_name='ap-northeast-2')
#     s3.put_object(
#         Bucket=S3_BUCKET,
#         Key=s3_key,
#         Body=json.dumps(data).encode('utf-8'),
#         ContentType='application/json',
#     )
#     logger.info(f'Saved metadata to s3://{S3_BUCKET}/{s3_key}')


def train_model(ti):
    base = ti.xcom_pull(task_ids='data_preparation', key='processed_s3_base')
    if not base:
        raise ValueError(f'No S3 base path received from data_preparation task')
    
    X_train = read_from_s3(f'{base}/X_train.parquet')
    X_test  = read_from_s3(f'{base}/X_test.parquet')
    y_train = read_from_s3(f'{base}/y_train.parquet').squeeze()
    y_test  = read_from_s3(f'{base}/y_test.parquet').squeeze()

    # Convert timestamp to datetime
    # X_train['timestamp'] = pd.to_datetime(X_train['timestamp'])
    # X_test['timestamp'] = pd.to_datetime(X_test['timestamp'])
    # y_train['timestamp'] = pd.to_datetime(y_train['timestamp'])
    # y_test['timestamp'] = pd.to_datetime(y_test['timestamp'])
    
    # # Set timestamp as index
    # X_train.set_index('timestamp',inplace=True)
    # X_test.set_index('timestamp',inplace=True)
    # y_train.set_index('timestamp',inplace=True)
    # y_test.set_index('timestamp',inplace=True)

    # MLflow setup
    # tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    # if not tracking_uri:
    #     logger.error(f'MLFLOW_TRACKING_URI not set in .env')
    #     raise ValueError(f'MLFLOW_TRACKING_URI not set in .env')
    MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:80"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  
    mlflow.set_experiment("automated_weather_delivery_delay_prediction_seoul")  

    with mlflow.start_run():
        params = {
            "n_estimators":339,
            "max_depth":17,
            "min_samples_split":9,
            "min_samples_leaf":1,
            "max_features":"sqrt",
            "random_state":42
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate (RMSE on log scale)
        y_pred_log = model.predict(X_test)
        y_pred_original = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_param("data_timestamp_min", str(X_train.index.min()))
        mlflow.log_param("data_timestamp_max", str(X_train.index.max()))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_dict(
            dict(zip(X_train.columns, model.feature_importances_)),
            "feature_importances.json"
        )
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Push to XCom
        run_id = mlflow.active_run().info.run_id
        ti.xcom_push(key="run_id", value=run_id)
        ti.xcom_push(key="rmse", value=rmse)