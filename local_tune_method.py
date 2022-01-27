import talib as ta
import numpy as np
import pandas as pd
import xgboost as xgb
import gc

################################################################get feature functions
def log_return(series, periods=5):
    return np.log(series).diff(periods)

def beta_resid(df, window): 
    num, unit = int(window[:-1]),window[-1]
    if unit == 'h':
        width = 60*num
    elif unit == 'd':
        width = 60*24*num
    b = ((ta.MULT(df.Mkt_lrt_15,df.lr_15).rolling(width).mean())/ \
        (ta.MULT(df.Mkt_lrt_15,df.Mkt_lrt_15).rolling(width).mean())).rename(f"beta_{window}")
    b = b.replace([np.nan,np.inf,-np.inf], 0)
    resids = ta.SUB(df.lr_15, ta.MULT(b, df.Mkt_lrt_15)).rename(f"lr_15_resid_{window}")
    return pd.concat([b, resids],axis=1)

def lag_features(df,fastk1,fastk2,adx,macd_s,macd_l,macd_sig,rsi,vol_sum,std_Crypto_Index,std_lr_15,std_Mkt_lrt_15,**kwargs):    
    ####TECH indicators
    df['slowK'], df['slowD'] = ta.STOCH(df.High, df.Low, df.Close, 
                                        fastk_period=fastk1, slowk_period=int(3*fastk1/5), slowd_period=int(3*fastk1/5),
                                        slowk_matype=0, slowd_matype=0)
    df['fastK'], df['fastD'] = ta.STOCHF(df.High, df.Low, df.Close,
                                         fastk_period=fastk2, fastd_period=int(3*fastk2/5), 
                                         fastd_matype=0)
    df[f'rsi_{rsi}'] = ta.RSI(df['Close'], timeperiod=rsi)
    df[f'macd_{macd_s}_{macd_l}'],df[f'macd_signal_{macd_sig}'], df['macd_hist'] = \
                ta.MACD(df['Close'],fastperiod=macd_s, slowperiod=macd_l, signalperiod=macd_sig)
    df[f'adx_{adx}'] = ta.ADX(df['High'], df['Low'],df['Close'], timeperiod=adx)#Average Directional Movement Index
    df[f'vol_sum_{vol_sum}'] = ta.SMA(df['Volume'],vol_sum)*vol_sum
    ####std volatility
    df[f'std_lr_15_{std_lr_15}'] = ta.STDDEV(df.lr_15,timeperiod=std_lr_15, nbdev=1)
    df[f'std_Mkt_lrt_15_{std_Mkt_lrt_15}'] = ta.STDDEV(df.Mkt_lrt_15,timeperiod=std_Mkt_lrt_15, nbdev=1)
    df[f'std_Crypto_Index_{std_Crypto_Index}'] = ta.STDDEV(df.Crypto_Index,timeperiod=std_Crypto_Index, nbdev=1)



def get_features(df_feat, fpara_dict):
    pd.options.mode.chained_assignment = None  # default='warn'
    ##beta,resids
    df_feat[[f"beta_{fpara_dict['beta_s']}",f"lr_15_resid_{fpara_dict['beta_s']}"]] = beta_resid(df_feat, window = fpara_dict['beta_s'])
    df_feat[[f"beta_{fpara_dict['beta_l']}",f"lr_15_resid_{fpara_dict['beta_l']}"]] = beta_resid(df_feat, window = fpara_dict['beta_l'])
    ##index lrtn
    df_feat[f"lrtn_index_{fpara_dict['lrtn']}"] = log_return(df_feat.Crypto_Index, fpara_dict['lrtn'])
    lag_features(df_feat, **fpara_dict)
    return df_feat
####################################################################################xgb_params_place holder
params_general ={'booster': 'gbtree', 'verbosity':0, 'validate_parameters': 1}
params_booster ={
    'learning_rate': 0.3,#check
    'gamma': 0, #gamma. check
    'max_depth': 6,#check
    'min_child_weight': 1, #instance weight (hessian). check
    'subsample': 0.8,#check
    'colsample_bytree': 1,#check
    'reg_lambda': 1,#L2 regularization term on weights
    'reg_alpha': 0, #L1 regularization term on weights
    'tree_method': 'hist', #hist, gpu_hist
    'predictor': 'auto', #auto, gpu_predictor
}
params_learning={
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'base_score':0.5,
    'seed': 2021
}

params_xgb = {**params_general, **params_booster, **params_learning}
params_train={
    'num_boost_round': 500, #alias as 'n_estimators' in sklearn api
    'early_stopping_rounds':10, 'verbose_eval': False
}
###########################################################################################cv
def make_cv_data(df_retrain, asset_id, psets):
    pd.options.mode.chained_assignment = None  # default='warn'
    tune_train = df_retrain[df_retrain["Asset_ID"] == asset_id]
    tune_train = get_features(tune_train,psets)
    tune_train.dropna(axis = 0, inplace= True)#for lag_features missing rows:<100
    dtrain=xgb.DMatrix(tune_train.drop(['timestamp', 'Asset_ID','Target','Weight'],axis=1),
                    label= tune_train['Target'])
    return dtrain

def CVrmse(dtrain, params_xgb, params_train):
    cvresult = xgb.cv(params_xgb, dtrain, 
                          num_boost_round = params_train['num_boost_round'], 
                          nfold=5, 
                          metrics=['rmse'], 
                          early_stopping_rounds = params_train['early_stopping_rounds'],
                          verbose_eval=False,as_pandas=True)
    return cvresult['test-rmse-mean'].min()
############################################################################################model fit
ASSET_DETAILS_CSV = './data/asset_details.csv'
df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")

def train_model_for_asset(df_train,asset_id, params_xgb,params_train, fpara_dict):
    pd.options.mode.chained_assignment = None
    dftrain = df_train[df_train["Asset_ID"] == asset_id].copy()
    dftrain = get_features(df_feat=dftrain, fpara_dict=fpara_dict)
    dftrain.dropna(axis = 0, inplace= True)
    dmat_train=xgb.DMatrix(data = dftrain.drop(['timestamp', 'Asset_ID','Target','Weight'],
                                        axis=1),
                    label= dftrain['Target'])
    del dftrain
    gc.collect()
    model = xgb.train(params_xgb, dtrain=dmat_train, 
                    evals=[(dmat_train,'train')],
                    **params_train)
    return model

def model_all_train(df_train, params_xgb,params_train,fpara_dict):
    models={}
    for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
        models[asset_id] = train_model_for_asset(df_train,asset_id, params_xgb,params_train, fpara_dict)
        print(f"finished model for asset_id {asset_id}", end="\r")
    return models
############################################################################################on-test score
def make_testset(df, fpara_dict):
    ###consistent timestamp for all 14 assets
    df2 = df.set_index("timestamp").copy()
    ind = df2.index.unique()
    def reindex(df):
        df = df.reindex(range(ind[0],ind[-1]+60,60),method='nearest')
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df
    df2 = df2.groupby('Asset_ID').apply(reindex).reset_index(0, drop=True).sort_index()
    ###add features
    df2 = df2.groupby('Asset_ID').apply(lambda x: get_features(x, fpara_dict))
    return df2.dropna(axis = 0).reset_index()

def weighted_correlation(a, b, weights):
  w = np.ravel(weights)
  a = np.ravel(a)
  b = np.ravel(b)
  sum_w = np.sum(w)
  mean_a = np.sum(a * w) / sum_w
  mean_b = np.sum(b * w) / sum_w
  var_a = np.sum(w * np.square(a - mean_a)) / sum_w
  var_b = np.sum(w * np.square(b - mean_b)) / sum_w

  cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
  corr = cov / np.sqrt(var_a * var_b)
  return corr

def test_score(df, models,fpara_dict):
    df_test = make_testset(df, fpara_dict)
    result_frame = []
    for id in range(0,14):
        model = models[id]
        x = df_test[df_test['Asset_ID']==id]
        x['Pred'] = model.predict(xgb.DMatrix(x[model.feature_names]))
        result_frame.append(x[['timestamp','Asset_ID','Weight','Target','Pred']])
    result = pd.concat(result_frame, axis=0)
    ########################################
    s=weighted_correlation(a=result['Target'], 
                        b=result['Pred'], 
                        weights=result['Weight'])
    return s

#################################################################final tune
def make_tune(df_retrain,df_test,params_xgb, feature_params, save_folder):
    dtrain = make_cv_data(df_retrain= df_retrain,asset_id=1, psets= feature_params)
    score_cv = CVrmse(dtrain,params_xgb,params_train)
    print(f"cv test-rmse: {score_cv}")
    
    print('Start train model on the sub-sample train set for scoring(9mins)')
    models = model_all_train(df_retrain, params_xgb, params_train, feature_params)
    
    score_on_test = test_score(df_test,models,feature_params)
    print(f"score-on-test: {score_on_test}")
    if score_on_test < 0.012:
        print("No improvement!!")
        return score_cv,score_on_test,None
    score_on_train =test_score(df_retrain[df_retrain['timestamp']>=df_retrain['timestamp'].quantile(0.75)],
                               models,feature_params)
    print(f"score-on-train: {score_on_train}")
    
    print('Start retrain model on the full data!!')
    df_train = pd.concat([df_retrain,df_test], join='outer')
    models_byalldata = model_all_train(df_train, params_xgb, params_train, feature_params)
    for asset_id in range(14):
        models_byalldata[asset_id].save_model(save_folder+ f'/model_{asset_id}.json')
    return score_cv,score_on_test,score_on_train