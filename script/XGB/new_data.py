import pandas as pd
import numpy as np
import gc

origin_train = pd.read_csv('./data/train.csv')
origin_sup = pd.read_csv('./data/supplemental_train.csv')

train = pd.concat([origin_train,origin_sup])
del origin_train, origin_sup
gc.collect()
print('data loaded')

train = train.set_index("timestamp").sort_index()
#clean VWAP
train.replace(to_replace=[np.inf, -np.inf],value= np.nan, inplace=True)
train['VWAP']=train['VWAP'].fillna((train['High']+train['Low'])/2)
#add weight
asset_detail = pd.read_csv('./data/asset_details.csv')
weight_dict=dict(zip(asset_detail.Asset_ID, asset_detail.Weight/asset_detail.Weight.sum()))
train['Weight'] = train['Asset_ID'].map(weight_dict)
train.set_index('Asset_ID',append=True, inplace=True)
#######################################add lr_15,mkt_lr_15,crypto_index
def log_return(series, periods=5):
    return np.log(series).diff(periods)

lr_15 = train.groupby('Asset_ID').apply( 
        lambda x: log_return(x[['Close']],15)
        )
train['lr_15'] = lr_15['Close']#same order:[timestamp, asset_id]
del lr_15
gc.collect()
print('lr_15 added')

mkt_lr_15 = train.groupby('timestamp').apply( 
    lambda x: x[["lr_15", "Close"]].multiply(x["Weight"], axis="index").sum()
    )
mkt_lr_15.columns = ['Mkt_lrt_15','Crypto_Index']#single index timestamp 
firsts = train.index.get_level_values('timestamp')
train[['Mkt_lrt_15','Crypto_Index']] = mkt_lr_15.loc[firsts].values
del mkt_lr_15
gc.collect()
print('Mkt_lrt_15,Crypto_Index added')

##fill in missing Target
from script.data_target import *
train = make_target(train)
train['Target'].fillna(train.Target2,inplace=True)
print('filling missing Target')
print(train.loc[train.Target.isin([np.nan, np.inf, -np.inf]),['Target','Target2']])


train.reset_index().drop(['R_15','Mkt_lrt','beta','Target2'],axis=1).to_feather('./data/new_data.ftr')
print('./data/new_data.ftr saved')