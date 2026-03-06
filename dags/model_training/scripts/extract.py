import io
import os
import boto3
import psycopg2
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

import logging

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


# Define a function to extract data with column names
def extract_data_with_columns(ti):
    # Get the connection
    
    hook = PostgresHook(postgres_conn_id='weather_connection')
    
    # Run the query
    sql = "SELECT * FROM feature_delays"
    try:
        result = hook.get_pandas_df(sql)
    except Exception as e:
        logger.error(f'Failed to execute query: {str(e)}')
        raise ValueError(f'Failed to execute query: {str(e)}')
    
    # Convert Timestamp or other non-serializable columns to strings
    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = result[col].astype(str)

    logger.info(f'result converted: {result}')
    # Alternatively, use raw cursor for more control
    # conn = hook.get_conn()
    # cursor = conn.cursor()
    # cursor.execute(sql)
    # rows = cursor.fetchall()
    # columns = [desc[0] for desc in cursor.description]
    # cursor.close()
    # conn.close()
    # result = {'columns': columns, 'data': rows}
    
    # save dataframe to S3
    logger.info(f'saving {len(result)} rows to S3')
    s3_key = f'model-training/{ti.run_id}/extract/raw.parquet'
    save_to_s3(result, s3_key)
    logger.info(f'Extraction complete — s3://{S3_BUCKET}/{s3_key}')
    ti.xcom_push(key='raw_s3_key', value=s3_key)  # push key, not data


    # Push results to XCom
    # ti.xcom_push(key='query_results', value={
    #     'columns': list(result.columns),
    #     'data': result.to_dict(orient='records')
    # })