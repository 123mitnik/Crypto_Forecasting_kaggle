'''
Given a dataframe,
Return train_df, val_df and test_df before normalization for the model
'''
import pandas as pd
from sklearn.model_selection import train_test_split

def mysplit(df,features, **split_para):
    df = df[features]
    train_df,val_df = train_test_split(df,train_size=split_para['train_size'],
                                                test_size=split_para['val_size'],shuffle=False)
    #holdout/test sets
    n = len(df) #rows
    test_start = int(n*(1-split_para['train_size']-split_para['val_size']))
    test_df = df[test_start:]

    return train_df, val_df, test_df


