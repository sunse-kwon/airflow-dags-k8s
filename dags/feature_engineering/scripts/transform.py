import numpy as np
import pandas as pd
import io
import boto3
import logging
# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BUCKET = 'feature-store-785685275217'


def save_to_s3(df: pd.DataFrame, s3_key: str):
    s3 = boto3.client('s3', region_name='ap-northeast-2')
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    buffer.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buffer.getvalue())
    logger.info(f'Saved features to s3://{S3_BUCKET}/{s3_key}')
    

def get_timestamp_index(data:pd.DataFrame)-> pd.DataFrame:
    data['base_time'] = data['base_time'].astype(str).apply(lambda x: x.zfill(4))
    data['timestamp'] = pd.to_datetime(data['base_date'].astype(str) + ' ' + data['base_time'], format='%Y-%m-%d %H%M')
    data.set_index('timestamp', inplace=True)
    return data


def generate_delay_hours(row):
    delay = 0
    if row['PTY'] >= 1: # if type of rainfall not 0(no rain), then add 30 min delay.
        delay += 0.5
    if row['RN1'] >= 1: # if rainfall more than 1mm, add 15min delay.
        delay += 0.25
    if row['WSD'] >= 7: # if wind speed higher than 7m/s, add 15 min delay.
        delay += 0.25
    if row['T1H'] < 0 or row['T1H'] > 30: # if temperature less than 0 or higher than 30 degree(extreme temperature), add 30min delay.
        delay += 0.5
    # limit maximum delay hour to 3 hour.
    delay = min(delay, 3.0)
    # add random noise
    delay += np.random.uniform(-0.2, 0.2)
    return max(0, delay) # prevent negative numbers.


def cleaning_data(data:pd.DataFrame, city_name:str) -> pd.DataFrame:
    data_pivot = data[data['city']==city_name].copy()
    data_common = data[data['city']==city_name].copy()

    data_common = data_common.groupby('timestamp').first()
    data_common = data_common[['base_date','year','month','day','day_of_week','is_holiday','base_time',
                               'hour','nx','ny','admin_district_code','city','sub_address']]

    data_pivot['timestamp'] = data_pivot.index
    data_pivot = data_pivot.pivot(index='timestamp', columns='category_code', values='measurement_value')

    data_pivot['delay_hours'] = data_pivot.apply(generate_delay_hours, axis=1)
    data_merged = pd.merge(data_pivot, data_common, left_index=True, right_index=True, how='left')

    data_resampled = data_merged.resample('1h').first()

    # missing data cleansing 
    numeric_time_features = ['REH', 'RN1', 'T1H', 'UUU', 'VEC', 'VVV', 'WSD', 'delay_hours', 'base_time', 'hour']
    categorical_date_features = ['nx', 'ny', 'admin_district_code', 'city','sub_address','base_date', 'year', 'month', 'day', 'day_of_week', 'is_holiday']

    data_resampled[numeric_time_features] = data_resampled[numeric_time_features].interpolate(method='time')
    data_resampled[categorical_date_features] = data_resampled[categorical_date_features].interpolate(method='bfill')

    data_resampled['PTY'] = data_resampled['PTY'].astype('category')
    data_resampled['PTY'].fillna(data_resampled['PTY'].mode()[0], inplace=True)

    data_resampled.drop(['base_time','sub_address','UUU', 'VEC', 'VVV'], axis=1, inplace=True)
    return data_resampled


def feature_engineering(data:pd.DataFrame) -> pd.DataFrame:
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)

    data['is_weekend'] = data['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    data['day_of_week_encoded'] = pd.Categorical(data['day_of_week']).codes

    # Convert is_holiday from TEXT 'TRUE'/'FALSE' to integers
    if 'is_holiday' in data.columns:
        data['is_holiday'] = data['is_holiday'].replace({'TRUE': 1, 'FALSE': 0}).astype(int)
    # data['is_holiday'] = data['is_holiday'].astype(int)
    
    data['PTY_lag1'] = data['PTY'].shift(1)
    data['PTY_lag2'] = data['PTY'].shift(2)

    data['delay_hours_lag1'] = data['delay_hours'].shift(1)
    data['delay_hours_lag2'] = data['delay_hours'].shift(2)
    
    data = data.fillna(method='bfill')
    return data

def transform_features(ti):

    data = ti.xcom_pull(task_ids='fetch_master_tables')
    logger.info("Step 1: set datetime index for time series")
    columns = [
        'measurement_value', 'base_date', 'year', 'month', 'day', 
        'day_of_week', 'is_holiday', 'base_time', 'hour', 'category_code', 
        'category_description', 'unit', 'nx', 'ny', 'admin_district_code', 
        'city', 'sub_address'
        ]
    data = pd.DataFrame(data, columns=columns)
    data = get_timestamp_index(data)

    transformed_features=[]
    for city in data.city.unique():
        try:
            logger.info(f'Step 2: data cleaning for {city}')
            cleaned_city = cleaning_data(data, city)

            logger.info(f'Step 3: feature engineering for {city}')
            feature_added = feature_engineering(cleaned_city)
       
            transformed_features.append(feature_added)
        except Exception as e:
            logger.error(f'failed transformation for {city}, see : {str(e)}')
    combined = pd.concat(transformed_features)
    s3_key = f'features/transformed/{ti.run_id}/features.parquet'
    save_to_s3(combined, s3_key)
    logger.info(f'Step 4: feature engineering done for all cities')
    ti.xcom_push(key='features_s3_key', value=s3_key)
    


def transform_features_hourly(ti):

    data = ti.xcom_pull(task_ids='fetch_master_tables')
    logger.info("Step 1: set datetime index for time series")
    columns = [
        'measurement_value', 'base_date', 'year', 'month', 'day', 
        'day_of_week', 'is_holiday', 'base_time', 'hour', 'category_code', 
        'category_description', 'unit', 'nx', 'ny', 'admin_district_code', 
        'city', 'sub_address'
        ]
    data = pd.DataFrame(data, columns=columns)
    data = get_timestamp_index(data)

    transformed_features=[]
    for city in data.city.unique():
        try:
            logger.info(f'Step 2: data cleaning for {city}')
            cleaned_city = cleaning_data(data, city)

            logger.info(f'Step 3: feature engineering for {city}')
            feature_added = feature_engineering(cleaned_city)
            # to select only current hour value. previous 2 hour data is not needed after create lag variable.
            feature_added = feature_added.iloc[-1:,:]
            transformed_features.append(feature_added)
        except Exception as e:
            logger.error(f'failed transformation for {city}, see : {str(e)}')
    
    combined = pd.concat(transformed_features)
    s3_key = f'features/hourly/{ti.run_id}/features.parquet'
    save_to_s3(combined, s3_key)

    logger.info('Step 4: feature engineering done for all cities')
    ti.xcom_push(key='features_s3_key', value=s3_key)