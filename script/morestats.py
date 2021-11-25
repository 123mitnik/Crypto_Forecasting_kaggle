from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
import re
import time
from datetime import date, datetime

######################basics
def s2d(secondint):
    '''
    convert the seconds(1970-01-01 00:00:00.000 UTC) format to datetime format.
    secondint can be a list-like input.
    return a datetime64[s]
    '''
    if isinstance(secondint, np.integer):
        t = secondint.astype('datetime64[s]')
    else:
        t = np.int64(secondint).astype('datetime64[s]')
    return t
def d2s(date_str):
    '''
    date_str format:"%Y-%m-%d"
    '''
    if isinstance(date_str, str):
        # given no seconds
        t = np.int32(time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple()))
    else:
        t = [np.int64(pd.Timestamp(s).timestamp()) for s in date_str]
    return t
def log_return(series, periods=1):
    '''
    log returns
    '''
    return np.log(series).diff(periods=periods)

def hist_volatility(series,window):
    '''
    moving historical standard deviation of the log returns
    '''
    return (log_return(series, periods=1).rolling(window).std()) * np.sqrt(window)
######################frequency converter
def ts_with_frequency(series, frequency='1min'):
    '''
    series is given as a minute-by-minute data with index as 'timestamp'.
    ts_frequency(): organize the series with required frequency(1min,5min,15min,1hour,1D)
    (by average over the smallest frequency:1minute)
    '''
    if series.index.name != 'timestamp':
        raise ValueError(f"The series is not with index named 'timestamp'")
    freq_valid_format = ['1min','5min','15min','1H','1D']
    if frequency not in freq_valid_format:
        raise ValueError(f"Wrong input: frequency should be a string as: {freq_valid_format} ")
    freq_num = int(re.findall(r'[0-9]+',frequency)[0]) #1,5,15
    freq_type = re.findall(r"\D+",frequency)[0] #min,H,D,W,M
    #add missing minute data as NAN:don’t fill gaps
    series = series.reindex(range(series.index[0],series.index[-1]+60,60),method='pad')
    #take average to fill in the window
    series.index = s2d(secondint=series.index)
    #The object must have a datetime-like index, average window start from current time
    series = series.resample(rule = frequency,label='right').mean()
    series.index = d2s(date_str=series.index)
    return series

def ohlcv_with_frequency(df, frequency='1min'):
    '''
    df is a dataframe with timestamp + ohlcv columns
    Count	Open	High	Low	Close	Volume	VWAP	
    '''
    #set time index
    df = df.set_index("timestamp")
    if df.index.name != 'timestamp':
        raise ValueError(f"The series is not with index named 'timestamp'")
    freq_valid_format = ['1min','5min','15min','1H','1D']
    if frequency not in freq_valid_format:
        raise ValueError(f"Wrong input: frequency should be a string as: {freq_valid_format} ")
    freq_num = int(re.findall(r'[0-9]+',frequency)[0]) #1,5,15
    freq_type = re.findall(r"\D+",frequency)[0] #min,hour,D,W,M
    #add missing minute data as NAN:don’t fill gaps
    df = df.reindex(range(df.index[0],df.index[-1]+60,60))
    df.index = s2d(secondint=df.index)
    #The object must have a datetime-like index, average window start from current time
    df = df.resample(rule = frequency,label='right').agg({'Count':'sum','Volume':'sum',
                                                    'High':'max', 'Low':'min',
                                                    'Open':'first','Close':'last',
                                                    'VWAP':'mean', 'Target':'last'})
    df.index = d2s(date_str=df.index)
    return df    
################time series decomposition
def ts_decomp(series, method):
    pass

################test
if __name__ == "__main__":
    btc = pd.read_csv('./codetest/btc.csv')
    print(btc.head())
    result = s2d(secondint=btc.timestamp)
    print(result)
    result = d2s(date_str=result)
    print(result[0:3])
    print(d2s(date_str='2018-09-09'))

    result = ts_with_frequency(series=btc.set_index('timestamp')['VWAP'],frequency='5min')
    #print(result.head())
    result = ohlcv_with_frequency(df=btc,frequency='5min')
    #print(result.head())
