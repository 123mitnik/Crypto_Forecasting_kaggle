import xgboost as xgb
from lightgbm import LGBMRegressor
import lightgbm as lgb
import numpy as np
import gc

######################################################initial model train setting
# Two new features from the competition tutorial
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
def get_features(df_feat):
    ##keep  timestamp
    #df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat

def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    # TODO: Try different features here!
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    del df_proc
    gc.collect()
    #Xgboost with Scikit-learn API
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=11,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        missing=-999,
        random_state=2020,
        # tree_method='gpu_hist'  # THE MAGICAL PARAMETER
    )
    model.fit(X, y)

    return X, y, model
######################################################new model train setting
import talib as ta
import pandas as pd

def log_return(series, periods=5):
    return np.log(series).diff(periods=periods)

def add_features(df):
    
    df["high_div_low"] = df["High"] / df["Low"]
    df["open_sub_close"] = df["Open"] - df["Close"]
    
    df['Open_shift-1'] = df['Open'].shift(-1)
    df['Open_shift-4'] = df['Open'].shift(-4)
    df['Open_shift-7'] = df['Open'].shift(-7)
    df['Open_shift1'] = df['Open'].shift(1)
    
    
    df['close_log1'] = log_return(df['Close'],periods=1)
    df['close_log5'] = log_return(df['Close'],periods=4)
    
    df['ema']   =ta.EMA(df['Open'], timeperiod=20)
    
    df['willr'] = ta.WILLR(df['High'], df['Low'],df['Close'], timeperiod=14)
    ##time info for dependent features
    times = pd.to_datetime(df["timestamp"],unit="s",infer_datetime_format=True)
    df["hour"] = times.dt.hour  
    df["dayofweek"] = times.dt.dayofweek 
    df["day"] = times.dt.day 
    
    df['rsi'] = ta.RSI(df['Close'], timeperiod=14)
    df['macd'], df['macdsignal'], df['MACD_HIST'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    df['u_band'], df['m_band'], df['l_band'] = ta.BBANDS(df['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['adx'] = ta.ADX(df['High'], df['Low'],df['Close'],timeperiod=14)
    df['DI_plus'] = ta.PLUS_DI(df['High'], df['Low'],df['Close'], timeperiod=14)
    df['DI_minus'] = ta.MINUS_DI(df['High'], df['Low'],df['Close'], timeperiod=14)
    #df = df.drop(['timestamp'],axis=1)
    df['ROCP'] =ta.ROCP(df['Open'])
    df['momentam'] =ta.MOM(df['Open'])
    df['APO'] =ta.APO(df['Open'])
    df['PPO'] =ta.PPO(df['Open'])
    df['CMO'] =ta.CMO(df['Open'])
    df['MIDPOINT'] =ta.MIDPOINT(df['Open'])
    df['TRENDLINE'] =ta.HT_TRENDLINE(df['Open'])
    return df

def get_Xy_and_model_for_asset_1(df_train,asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    df = df.dropna(subset=['Target'])
    y = df['Target'] 
    df = df.drop(['Target'],axis=1)
    df_proc = get_features(df)
    df_proc = add_features(df_proc)
    df_proc = df_proc.fillna(-1)
    df_proc = df_proc.drop(['timestamp', 'Asset_ID','hour','dayofweek', 'day'],axis=1)
    X= df_proc #.drop("y", axis=1)
    #print(f'features:{X.columns}')
    del df_proc
    del df
    gc.collect()

    model = LGBMRegressor(n_estimators=1000,num_leaves=500,learning_rate=0.1)
    model.fit(X, y)

    return X,y, model