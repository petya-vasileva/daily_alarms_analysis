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
