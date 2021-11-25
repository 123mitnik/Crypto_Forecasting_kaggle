import numpy as np

# A utility function to build features from the original df
def get_features(df):
    df['Upper_Shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['Lower_Shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']
    df['spread'] = df['High'] - df['Low']
    df['mean_trade'] = df['Volume']/df['Count']
    df['log_price_change'] = np.log(df['Close']/df['Open'])
    return df

