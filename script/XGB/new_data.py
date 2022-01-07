import pandas as pd
import numpy as np
import datatable as dt
import os

# WHICH YEARS TO INCLUDE
INC2021 = 1
INC2020 = 1
INC2019 = 1
INC2018 = 0
INC2017 = 0
INCCOMP = 0
INCSUPP = 0

#paths
data_path = '/Users/dingxian/Documents/GitHub/Crypto_Forecasting_kaggle/data'
extra_data_files = {0: data_path+'/cryptocurrency-extra-data-binance-coin', 
                    2: data_path+'/cryptocurrency-extra-data-bitcoin-cash', 
                    1: data_path+'/cryptocurrency-extra-data-bitcoin', 
                    3: data_path+'/cryptocurrency-extra-data-cardano', 
                    4: data_path+'/cryptocurrency-extra-data-dogecoin', 
                    5: data_path+'/cryptocurrency-extra-data-eos-io', 
                    6: data_path+'/cryptocurrency-extra-data-ethereum', 
                    7: data_path+'/cryptocurrency-extra-data-ethereum-classic', 
                    8: data_path+'/cryptocurrency-extra-data-iota', 
                    9: data_path+'/cryptocurrency-extra-data-litecoin', 
                    11: data_path+'/cryptocurrency-extra-data-monero', 
                    10: data_path+'/cryptocurrency-extra-data-maker', 
                    12: data_path+'/cryptocurrency-extra-data-stellar', 
                    13: data_path+'/cryptocurrency-extra-data-tron'}
print('loading preparing')

def load_training_data_for_asset(asset_id, load_jay = True, includeextra=True):
    dfs = []
    # original data
    if INCCOMP: 
        orig_df_train = dt.fread(data_path+'/cryptocurrency-extra-data-binance-coin/orig_train.jay').to_pandas()
        dfs.append(orig_df_train[orig_df_train["Asset_ID"] == asset_id].copy())
    if INCSUPP: 
        supp_df_train = dt.fread(data_path+'/cryptocurrency-extra-data-binance-coin/orig_supplemental_train.jay').to_pandas()
        dfs.append(supp_df_train[supp_df_train["Asset_ID"] == asset_id].copy())
    
    if includeextra:
        #extra data
        if load_jay:
            if INC2017 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2017) + '.csv'): 
                dfs.append(dt.fread(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2017) + '.jay').to_pandas())
            if INC2018 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2018) + '.csv'): dfs.append(dt.fread(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2018) + '.jay').to_pandas())
            if INC2019 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2019) + '.csv'): dfs.append(dt.fread(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2019) + '.jay').to_pandas())
            if INC2020 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2020) + '.csv'): dfs.append(dt.fread(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2020) + '.jay').to_pandas())
            if INC2021 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2021) + '.csv'): dfs.append(dt.fread(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2021) + '.jay').to_pandas())
        else: 
            if INC2017 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2017) + '.csv'): dfs.append(pd.read_csv(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2017) + '.csv'))
            if INC2018 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2018) + '.csv'): dfs.append(pd.read_csv(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2018) + '.csv'))
            if INC2019 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2019) + '.csv'): dfs.append(pd.read_csv(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2019) + '.csv'))
            if INC2020 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2020) + '.csv'): dfs.append(pd.read_csv(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2020) + '.csv'))
            if INC2021 and os.path.exists(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2021) + '.csv'): dfs.append(pd.read_csv(extra_data_files[asset_id] + '/full_data__' + str(asset_id) + '__' + str(2021) + '.csv'))
    df = pd.concat(dfs, axis = 0) if len(dfs) > 1 else dfs[0]
    df['date'] = pd.to_datetime(df['timestamp'], unit = 's')
    df = df.sort_values('date')
    return df

def load_data_for_all_assets(load_jay = True, includeextra=True):
    dfs = []
    for asset_id in list(extra_data_files.keys()): 
        dfs.append(load_training_data_for_asset(asset_id, load_jay = load_jay, includeextra=includeextra))
    train = pd.concat(dfs)
    return train[['timestamp','Asset_ID','Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']]

train = load_data_for_all_assets(load_jay = True, includeextra=True)
print(train.head())

#check nan
for i in range(14):
    dfcrop=train[train['Asset_ID']==i]
    print('Percentage of values not nan',(1-(np.sum((dfcrop['Target'].isnull()).astype(int))/dfcrop.shape[0]))*100)

print('saving data')
train.reset_index().to_feather(data_path+'/new_data.ftr')
#print(pd.read_feather(data_path+'/new_data.ftr'))

#check time range
print(pd.to_datetime(train["timestamp"],unit="s",infer_datetime_format=True))



    