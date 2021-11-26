import pandas as pd
import numpy as np
import datatable as dt
import os

# WHICH YEARS TO INCLUDE
INC2021 = 1
INC2020 = 1
INC2019 = 0
INC2018 = 0
INC2017 = 0
INCCOMP = 0
INCSUPP = 0
if __name__ != '__main__':
    input = [INC2021,INC2020,INC2019,INC2018 ,INC2017 ,INCCOMP ,INCSUPP]
    y =['INC2021','INC2020','INC2019','INC2018' ,'INC2017' ,'INCCOMP' ,'INCSUPP']
    for s,i in zip(y,input):
        if i:
            print(f'select years {s}')
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
# orig_df_train = pd.read_csv(data_path + '/train.csv') 
# supp_df_train = pd.read_csv(data_path + '/supplemental_train.csv')
#faster with .jay
orig_df_train = dt.fread(data_path+'/cryptocurrency-extra-data-binance-coin/orig_train.jay').to_pandas()
supp_df_train = dt.fread(data_path+'/cryptocurrency-extra-data-binance-coin/orig_supplemental_train.jay').to_pandas()

def load_training_data_for_asset(asset_id, load_jay = True, includeextra=True):
    dfs = []
    # original data
    if INCCOMP: dfs.append(orig_df_train[orig_df_train["Asset_ID"] == asset_id].copy())
    if INCSUPP: dfs.append(supp_df_train[supp_df_train["Asset_ID"] == asset_id].copy())
    
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
    train[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']] = train[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']].astype(np.float32)
    return train

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ 
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name
        
        if col_type not in ['object', 'category', 'datetime64[ns, UTC]', 'datetime64[ns]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df