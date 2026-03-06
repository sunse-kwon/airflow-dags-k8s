import io
import os
import boto3
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

random.seed(42)         
np.random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BUCKET = 'feature-store-785685275217'


def save_to_s3(df: pd.DataFrame, s3_key: str):
    s3 = boto3.client('s3', region_name='ap-northeast-2')
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    buffer.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buffer.getvalue())
    logger.info(f'Saved to s3://{S3_BUCKET}/{s3_key}')

def read_from_s3(s3_key: str) -> pd.DataFrame:
    s3 = boto3.client('s3', region_name='ap-northeast-2')
    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buffer)
    buffer.seek(0)
    return pd.read_parquet(buffer)


def prepare_data(ti):
   s3_key_in = ti.xcom_pull(task_ids='data_extraction', key='raw_s3_key')

   if not s3_key_in:
      logger.error(f'No S3 key received from data_extraction task')
      raise ValueError(f'No S3 key received from data_extraction task')
   
   logger.info('reading raw data from S3')
   data = read_from_s3(s3_key_in)
   logger.info(f'Loaded {len(data)} rows — sample:\n{data.head()}')

   # # Example: Convert to DataFrame
   # data = pd.DataFrame(query_result['data'], columns=query_result['columns'])
   # logger.info(f'first data from previous task: {data.head()}')

   # target log transformation
   data['delay_hours_log'] = np.log1p(data['delay_hours'])

   # filter Seoul fulfillment center time series only. after implementation, scale out to other 3 regions (Incheon, Daegu, Icheon)
   data_seoul = data[data['city']=='Seoul'].copy()

   # Set timestamp as index
   data_seoul['timestamp'] = pd.to_datetime(data_seoul['timestamp'], format='%Y-%m-%d %H:%M:%S')  # Ensure datetime format
   data_seoul.set_index('timestamp', inplace=True)

   features=['pty','reh','rn1','t1h','wsd','day','hour','sin_hour','cos_hour','is_weekend',
       'day_of_week_encoded','pty_lag1','pty_lag2','delay_hours_lag1','delay_hours_lag2']
    
   X  = data_seoul[features]
   y = data_seoul['delay_hours_log']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

   logger.info(f'after split, check sample X_train: {X_train.head()}')


   logger.info('saving splits to S3')
   base = f'model-training/{ti.run_id}/processed'

   save_to_s3(X_train,         f'{base}/X_train.parquet')
   save_to_s3(X_test,          f'{base}/X_test.parquet')
   save_to_s3(y_train.to_frame(), f'{base}/y_train.parquet')
   save_to_s3(y_test.to_frame(),  f'{base}/y_test.parquet')

   ti.xcom_push(key='processed_s3_base', value=base)

#  # Reset index to include timestamp as a column
#    X_train = X_train.reset_index()  
#    X_test = X_test.reset_index()    
#    y_train = y_train.reset_index()  
#    y_test = y_test.reset_index()    

   # # Convert timestamp to string for JSON serialization
   # X_train['timestamp'] = X_train['timestamp'].astype(str)
   # X_test['timestamp'] = X_test['timestamp'].astype(str)
   # y_train['timestamp'] = y_train['timestamp'].astype(str)
   # y_test['timestamp'] = y_test['timestamp'].astype(str)

   # logger.info(f'last, check sample X_test: {X_test.head()}')

   # # Push to XCom
   # ti.xcom_push(key="X_train", value=X_train.to_dict(orient='records'))
   # ti.xcom_push(key="X_test", value=X_test.to_dict(orient='records'))
   # ti.xcom_push(key="y_train", value=y_train.to_dict(orient='records'))
   # ti.xcom_push(key="y_test", value=y_test.to_dict(orient='records'))
    