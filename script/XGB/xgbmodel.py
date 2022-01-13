import xgboost as xgb
#from lightgbm import LGBMRegressor
#import lightgbm as lgb

import numpy as np
import gc
import talib as ta
import pandas as pd
import time


######################################################feature engineering
def log_return(series, periods=5):
    return np.log(series).diff(periods=periods)

def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']    

def rsi(df, periods = 15, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    RSI forecasts an upcoming reversal of a trend.
    """
    close_delta = df['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema:
	    # Use exponential moving average, com=dacay
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods).mean()
        ma_down = down.rolling(window = periods).mean()
    ##inf issue
    ma_up.loc[ma_up==ma_down] =1
    ma_down.loc[ma_up==ma_down] =1
    ma_down.loc[ma_down==0] =0.000001
    
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def macd(df,period_l=30,period_s=15):
    '''
     trend-following momentum
    '''
    ema1=df['Close'].ewm(span=period_s, adjust=False).mean()
    ema2=df['Close'].ewm(span=period_l, adjust=False).mean()
    return ema1-ema2 

def get_sma(prices, window):
    return prices.rolling(window).mean()

def get_bollinger_bands(prices, window=15):
    sma = get_sma(prices, window)
    std = prices.rolling(window).std()
    bollinger_up = sma + std * 2 # Calculate top band
    bollinger_down = sma - std * 2 # Calculate bottom band
    return bollinger_up, bollinger_down
    
def lag_features(df):
    #Close-log-return
    df['lrtn_close_5'] = log_return(df['Close'],periods=5)
    df['lrtn_close_15'] = log_return(df['Close'],periods=15)
    df['lrtn_index_5'] = log_return(df['Crypto_Index'],periods=5)
    df['lrtn_index_15'] = log_return(df['Crypto_Index'],periods=15)
    #15minutes-volume-sum/delta, on-balance-volume
    df['vol_sum_15'] = df['Volume'].rolling(window=15).sum()
    df['vol_delta_15'] = df['vol_sum_15'].diff()
    df['vol_pressure_1']=np.sign(df['Close'].diff(1)) * df['Volume']
    df['vol_pressure_15']=np.sign(df['Close'].diff(15)) * df['vol_sum_15']
    #tech analysis indicators
    df['rsi_15'] = rsi(df,periods=15,ema = False)
    df['macd_15_30'] = macd(df,period_l=30,period_s=15)
    band_up,band_down=get_bollinger_bands(df['Close'],window=15)
    df['close_bollinger_up_15'] = band_up - df['Close']
    df['close_bollinger_down_15'] = df['Close'] - band_down

def get_features(df_feat, lagfeatures=False):
    pd.options.mode.chained_assignment = None  # default='warn'
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    if lagfeatures:
        lag_features(df_feat)
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
####################################################################################parameters
params_general ={
    'booster': 'gbtree',
    'verbosity':0,
    'validate_parameters': 1
}
params_booster ={
    'learning_rate': 0.05,
    'min_split_loss': 0,
    'max_depth': 11,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'reg_lambda': 1,
    'reg_alpha': 0,
    'tree_method': 'hist', #gpu_hist
    'scale_pos_weight': 1,
    'predictor': 'auto',
    'num_parallel_tree': 1
}

params_learning={
    'objective': 'reg:squarederror',
    'base_score': 0.5,
    'eval_metric': 'rmse',
    'seed': 2021
}

params_train={
    'num_boost_round':500,
    'early_stopping_rounds':5,
    'verbose_eval':False
}

params_xgb = {**params_general,
            **params_booster,
            **params_learning}

def get_Xy_and_model_for_asset(df_train, asset_id):
    '''
    XGBoost
    '''
    pd.options.mode.chained_assignment = None  # default='warn'
    df = df_train[df_train["Asset_ID"] == asset_id]
    df = get_features(df)
    df = add_features(df)
    df.dropna(axis = 0, inplace= True)
    dtrain=xgb.DMatrix(df.drop(['timestamp', 'Asset_ID','Target'],axis=1)
                        , label= df['Target'])
    del df
    gc.collect()

    start_time = time.time()
    #Xgboost with Learning API
    ##TODO make evals at test data
    model = xgb.train(params_xgb, dtrain=dtrain, evals=[(dtrain, 'train')], **params_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    return model

######################################################lightgbm_model
def get_Xy_and_model_for_asset_1(df_train,asset_id):
    '''
    lightgbm
    '''
    pass
    # df = df_train[df_train["Asset_ID"] == asset_id]
    # df = df.dropna(subset=['Target'])
    # y = df.pop('Target') 
    # df_proc = get_features(df)
    # del df
    # df_proc = add_features(df_proc)
    # df_proc = df_proc.fillna(-1)
    # X = df_proc.drop(['timestamp', 'Asset_ID'],axis=1)
    # #print(f'features:{X.columns}')
    # del df_proc
    # gc.collect()

    # #lightxgboost
    # model = LGBMRegressor(n_estimators=1000,num_leaves=500,learning_rate=0.1)
    # model.fit(X, y)

    # return model