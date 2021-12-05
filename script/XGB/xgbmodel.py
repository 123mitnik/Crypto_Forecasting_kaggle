import xgboost as xgb
from lightgbm import LGBMRegressor
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import gc
import talib as ta
import pandas as pd
import time


######################################################initial model train setting
# Two new features from the competition tutorial
def log_return(series, periods=5):
    return np.log(series).diff(periods=periods)

def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
def get_features(df_feat):
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat
def add_features(df, usetalib=True):
    df['lrtn_1'] = log_return(df['Close'],periods=1)
    df['lrtn_4'] = log_return(df['Close'],periods=4)

    if usetalib:
        df['ema']   =ta.EMA(df['Open'], timeperiod=20)
        df['willr'] = ta.WILLR(df['High'], df['Low'],df['Close'], timeperiod=14)
        ##time info for dependent features
        df['rsi'] = ta.RSI(df['Close'], timeperiod=14)
        df['macd'], df['macdsignal'], df['MACD_HIST'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        df['u_band'], df['m_band'], df['l_band'] = ta.BBANDS(df['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        df['adx'] = ta.ADX(df['High'], df['Low'],df['Close'],timeperiod=14)
        df['DI_plus'] = ta.PLUS_DI(df['High'], df['Low'],df['Close'], timeperiod=14)
        df['DI_minus'] = ta.MINUS_DI(df['High'], df['Low'],df['Close'], timeperiod=14)
        df['ROCP'] =ta.ROCP(df['Open'])
        df['momentam'] =ta.MOM(df['Open'])
        df['APO'] =ta.APO(df['Open'])
        df['PPO'] =ta.PPO(df['Open'])
        df['CMO'] =ta.CMO(df['Open'])
        df['MIDPOINT'] =ta.MIDPOINT(df['Open'])
        df['TRENDLINE'] =ta.HT_TRENDLINE(df['Open'])
    return df

params_xgb = {
    'learning_rate': 0.05,
    'max_depth': 11,
    'n_estimators': 500,
    'subsample': 0.9,
    'colsample_bytree':0.7,
    'missing':-999,
    'random_state':2020,
    'objective':  'reg:squarederror'
}
def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    df = get_features(df)
    df = add_features(df)
    df.dropna(axis = 0, inplace= True)

    y = df.pop('Target') 
    X = df.drop(['timestamp', 'Asset_ID'],axis=1)
    del df
    gc.collect()

    start_time = time.time()
    #Xgboost with Scikit-learn API
    model = xgb.XGBRegressor(**params_xgb)
    model.fit(X, y)
    print("--- %s seconds ---" % (time.time() - start_time))
    return model

######################################################lightgbm_model
def get_Xy_and_model_for_asset_1(df_train,asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    df = df.dropna(subset=['Target'])
    y = df.pop('Target') 
    df_proc = get_features(df)
    del df
    df_proc = add_features(df_proc)
    df_proc = df_proc.fillna(-1)
    X = df_proc.drop(['timestamp', 'Asset_ID'],axis=1)
    #print(f'features:{X.columns}')
    del df_proc
    gc.collect()

    #lightxgboost
    model = LGBMRegressor(n_estimators=1000,num_leaves=500,learning_rate=0.1)
    model.fit(X, y)

    return X,y, model