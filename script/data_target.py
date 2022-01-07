'''
The target calculation is based on the close price of the asset 
and can be derived from the provided data using the methodology in 
https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition.
'''  
import numpy as np

def ResidualizeMarket(df, mktColumn, window):
  if mktColumn not in df.columns:
    return df

  mkt = df[mktColumn]

  num = df.multiply(mkt.values, axis=0).rolling(window).mean().values  #numerator of linear regression coefficient
  denom = mkt.multiply(mkt.values, axis=0).rolling(window).mean().values  #denominator of linear regression coefficient
  beta = np.nan_to_num( num.T / denom, nan=0., posinf=0., neginf=0.)  #if regression fell over, use beta of 0

  resultRet = df - (beta * mkt.values).T  #perform residualization
  resultBeta = 0.*df + beta.T  #shape beta

  return resultRet.drop(columns=[mktColumn]), resultBeta.drop(columns=[mktColumn])