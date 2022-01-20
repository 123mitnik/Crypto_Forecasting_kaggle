'''
The target calculation is based on the close price of the asset 
and can be derived from the provided data using the methodology in 
https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition.
'''  
import numpy as np
import talib as ta


def beta(df, window=3750): 
  '''
  linear regression R on M to generate beta and residuals
  '''
  b= (ta.MULT(df.Mkt_lrt, df.R_15).rolling(window).mean())/(ta.MULT(df.Mkt_lrt,df.Mkt_lrt).rolling(window).mean())
  return b

def make_target(dd):
  '''
  input dd: a dataframe with columns [lr_15,Mkt_lrt_15].
  '''
  dd.set_index(['timestamp','Asset_ID'],inplace=True)
  ##shift [lr_15,Mkt_lrt_15] backward by 16
  ddd=dd.groupby('Asset_ID').apply(lambda x: x[['lr_15','Mkt_lrt_15']].shift(-16))
  ddd.columns = ['R_15','Mkt_lrt']
  dd= dd.merge(ddd, on =['timestamp','Asset_ID'],how='left')
  ##make beta  
  ddd=dd.groupby("Asset_ID").apply(lambda x: beta(x)).rename("beta").to_frame().reset_index(0,drop=True)
  ddd=ddd.replace([np.nan,np.inf,-np.inf], 0)
  dd= dd.merge(ddd, on =['timestamp','Asset_ID'],how='left')
  ##make target
  dd['Target2']=ta.SUB(dd.R_15 , ta.MULT(dd.beta,dd.Mkt_lrt))
  return dd