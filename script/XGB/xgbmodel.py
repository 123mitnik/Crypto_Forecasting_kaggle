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
    return np.log(series).diff(periods)

def upper_shadow(df):
    return ta.SUB(df['High'], np.maximum(df['Close'], df['Open']))

def lower_shadow(df):
    return ta.SUB(np.minimum(df['Close'], df['Open']), df['Low'] )

def lag_features(df):
    #Close-log-return
    df['lrtn_close_5'] = log_return(df['Close'],periods=5)
    df['lrtn_close_15'] = log_return(df['Close'],periods=15)
    df['lrtn_index_5'] = log_return(df['Crypto_Index'],periods=5)
    df['lrtn_index_15'] = log_return(df['Crypto_Index'],periods=15)
    
    #15minutes-volume-sum/delta, on-balance-volume
    df['vol_sum_15'] = ta.SMA(df['Volume'],15)*15
    df['vol_delta_15'] = df['vol_sum_15'].diff()
    df['vol_pressure_1']=np.sign(df['Close'].diff(1)) * df['Volume']
    df['vol_pressure_15']=ta.MULT(np.sign(df['Close'].diff(15)), df['vol_sum_15'])
    
    #tech analysis indicators
    df['willr'] = ta.WILLR(df['High'], df['Low'],df['Close'], timeperiod=15)
    df['adx'] = ta.ADX(df['High'], df['Low'],df['Close'],timeperiod=15)
    df['DI_plus'] = ta.PLUS_DI(df['High'], df['Low'],df['Close'], timeperiod=14)
    df['DI_minus'] = ta.MINUS_DI(df['High'], df['Low'],df['Close'], timeperiod=14)
    df['ROCP'] =ta.ROCP(df['Open'])
    df['momentam'] =ta.MOM(df['Open'])
    df['APO'] =ta.APO(df['Open'])
    df['PPO'] =ta.PPO(df['Open'])
    df['CMO'] =ta.CMO(df['Open'])
    df['MIDPOINT'] =ta.MIDPOINT(df['Open'])
    df['TRENDLINE'] =ta.HT_TRENDLINE(df['Open'])
    
    df['rsi_15'] = ta.RSI(df['Close'], timeperiod=15)
    df['macd_15_30'],df['macd_signal'], df['macd_hist'] = ta.MACD(df['Close'], fastperiod=15, slowperiod=30, signalperiod=5)
    band_up, mid_band, band_down = ta.BBANDS(df['Close'], timeperiod=15, nbdevup=2, nbdevdn=2, matype=0)
    df['close_bollinger_up_15'] = ta.SUB(band_up, df['Close'])
    df['close_bollinger_down_15'] = ta.SUB(df['Close'], band_down)

def get_features(df_feat, lagfeatures=False):
    pd.options.mode.chained_assignment = None  # default='warn'
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    if lagfeatures:
        lag_features(df_feat)
    return df_feat
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