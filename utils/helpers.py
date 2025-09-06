from datetime import datetime, timedelta
import dateutil.relativedelta
import time
import os
import json
import pandas as pd
import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import getpass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools
from functools import wraps


INDICES = ['ps_packetloss', 'ps_owd',
           'ps_retransmits', 'ps_throughput', 'ps_trace']

# user, passwd, mapboxtoken = None, None, None
# with open('/config/config.json') as json_data:
#     config = json.load(json_data,)

# es = Elasticsearch(
#     hosts=[{'host': config['ES_HOST'], 'port':9200, 'scheme':'https'}],
#     http_auth=(config['ES_USER'], config['ES_PASS']),
#     request_timeout=60)



with open("creds.key") as f:
    user = f.readline().strip()
    passwd = f.readline().strip()
    mapboxtoken = f.readline().strip()

def ConnectES():
    global user, passwd
    credentials = (user, passwd)

    try:
        es = Elasticsearch('https://localhost:9200', verify_certs=False,
                           http_auth=credentials, max_retries=5, retry_on_timeout=True)
        print('Success' if es.ping()==True else 'Fail')
        return es
    except Exception as error:
        print (">>>>>> Elasticsearch Client Error:", error)

es = ConnectES()


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__} in {run_time:.4f} secs")
        return value
    return wrapper_timer


''' Takes a function, splits a dataframe into 
batches and excutes the function passing a batch as a parameter.'''
def parallelPandas(function):
    @wraps(function)
    def wrapper(dataframe):
        cores = 16
        splits = np.array_split(dataframe, cores*2)
        result = []

        with ProcessPoolExecutor(max_workers=cores) as pool:
            result.extend(pool.map(function, splits))

        frame = pd.DataFrame()
        for data in result:
            frame = pd.concat([frame, data])

        return frame
    return wrapper


def convertTime(ts):
    if pd.notnull(ts):
        return datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M')


'''Returns a period of the past 3 hours'''
def defaultTimeRange(hours=3):
    now = datetime.utcnow()
    defaultEnd = datetime.strftime(now, '%Y-%m-%d %H:%M')
    defaultStart = datetime.strftime(
        now - timedelta(hours=hours), '%Y-%m-%d %H:%M')

    return [defaultStart, defaultEnd]


'''Finds the difference between two dates'''
def FindPeriodDiff(dateFrom, dateTo):
    fmt = '%Y-%m-%d %H:%M'
    d1 = datetime.strptime(dateFrom, fmt)
    d2 = datetime.strptime(dateTo, fmt)
    time_delta = d2-d1

    return time_delta


'''Splits the period into chunks of specified number of intervals.'''
def GetTimeRanges(dateFrom, dateTo, intv=1):
    diff = FindPeriodDiff(dateFrom, dateTo) / intv
    t_format = "%Y-%m-%d %H:%M"
    tl = []
    for i in range(intv+1):
        t = (datetime.strptime(dateFrom, t_format) + diff * i).strftime(t_format)
        tl.append(int(time.mktime(datetime.strptime(t, t_format).timetuple())*1000))

    return tl


'''The following method helps to calculate the expected number of tests for a specific period'''
def CalcMinutes4Period(dateFrom, dateTo):
    time_delta = FindPeriodDiff(dateFrom, dateTo)

    return (time_delta.days*24*60 + time_delta.seconds//60)


def roundTime(dt=None, round_to=60*60):
    if dt == None:
        dt = datetime.utcnow()
    seconds = (dt - dt.min).seconds
    rounding = (seconds+round_to/2) // round_to * round_to
    return dt + timedelta(0, rounding-seconds, -dt.microsecond)


def split_time_period(start_str, end_str, bin_hours=12):
    # Parse the start and end times
    start_time = datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    end_time = datetime.strptime(end_str, '%Y-%m-%dT%H:%M:%S.%fZ')

    # Calculate the bin duration as a timedelta object
    bin_duration = timedelta(hours=bin_hours)

    # Initialize the list of bins
    bins = []

    # Initialize the current start time for binning
    current_start_time = start_time

    # Loop to create bins
    while current_start_time < end_time:
        # Calculate the current end time for the bin
        current_end_time = current_start_time + bin_duration

        # Ensure the current end time does not exceed the overall end time
        if current_end_time > end_time:
            current_end_time = end_time

        # Append the current bin as [start, end] in the specified format
        bins.append([
            current_start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            current_end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        ])

        # Update the start time for the next bin
        current_start_time = current_end_time

    return bins


def normalize_timestamp_column(df, timestamp_col='timestamp', datetime_col='dt', dt_str_col='dt_str', unit='s'):
    """
    Normalize timestamp column to consistent format for Parquet compatibility
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with timestamp data
    timestamp_col : str
        Name of the timestamp column to normalize (default: 'timestamp')
    datetime_col : str  
        Name of the datetime column to create (default: 'dt')
    dt_str_col : str
        Name of the human-readable datetime string column (default: 'dt_str')
    unit : str
        Unit for timestamp conversion ('s' for seconds, 'ms' for milliseconds)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized timestamp columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Normalize timestamp column to Int64 for Parquet compatibility
    if timestamp_col in df.columns:
        # Handle various timestamp formats
        def normalize_timestamp_value(ts):
            if pd.isna(ts) or ts is None:
                return pd.NA
            
            if isinstance(ts, str):
                try:
                    # Try parsing ISO string first
                    dt = pd.to_datetime(ts, utc=True)
                    if unit == 'ms':
                        return int(dt.timestamp() * 1000)
                    else:
                        return int(dt.timestamp())
                except:
                    # Try to parse as numeric string
                    try:
                        return int(float(ts))
                    except:
                        return pd.NA
            
            # Convert to numeric and return
            numeric_ts = pd.to_numeric(ts, errors='coerce')
            if pd.isna(numeric_ts):
                return pd.NA
            return int(numeric_ts)
        
        df[timestamp_col] = df[timestamp_col].apply(normalize_timestamp_value)
        df[timestamp_col] = df[timestamp_col].astype('Int64')
    
    # Create datetime column from normalized timestamps
    df[datetime_col] = pd.NaT
    
    valid_mask = df[timestamp_col].notna()
    if valid_mask.any():
        try:
            df.loc[valid_mask, datetime_col] = pd.to_datetime(
                df.loc[valid_mask, timestamp_col], 
                unit=unit, 
                utc=True
            )
            # Ensure datetime column is properly typed
            df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
        except Exception as e:
            print(f"Warning: Could not convert timestamps to datetime: {e}")
            print(f"Sample timestamp values: {df.loc[valid_mask, timestamp_col].head(3).tolist()}")
            print(f"Timestamp column dtype: {df[timestamp_col].dtype}")
            # Create properly typed empty datetime column
            df[datetime_col] = pd.to_datetime(pd.Series([pd.NaT] * len(df), dtype='datetime64[ns, UTC]'))
    
    # Create human-readable datetime string using safe function
    df[dt_str_col] = safe_datetime_to_string(df[datetime_col])
    
    return df


def safe_datetime_to_string(dt_series, format_str='%Y-%m-%d %H:%M:%S UTC'):
    """
    Safely convert datetime series to string, handling NaT values
    
    Parameters:
    -----------
    dt_series : pandas.Series
        Series with datetime values (may contain NaT)
    format_str : str
        Format string for datetime conversion
    
    Returns:
    --------
    pandas.Series
        Series with formatted datetime strings (None for NaT values)
    """
    if dt_series is None or dt_series.empty:
        return pd.Series([None] * len(dt_series), index=dt_series.index) if not dt_series.empty else pd.Series()
    
    result = pd.Series([None] * len(dt_series), index=dt_series.index, dtype='object')
    
    # Check if series has datetime-like values
    try:
        valid_mask = dt_series.notna()
        if valid_mask.any():
            # Ensure we have a proper datetime series
            dt_subset = dt_series.loc[valid_mask]
            if hasattr(dt_subset, 'dt'):
                result.loc[valid_mask] = dt_subset.dt.strftime(format_str)
            else:
                # Try to convert to datetime first
                dt_converted = pd.to_datetime(dt_subset, errors='coerce')
                valid_converted = dt_converted.notna()
                if valid_converted.any():
                    result.loc[valid_mask[valid_converted]] = dt_converted.loc[valid_converted].dt.strftime(format_str)
    except Exception as e:
        print(f"Warning: Could not format datetime strings: {e}")
        # Return series with None values
        pass
    
    return result
