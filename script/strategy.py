import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
Moving average crossover
https://www.datacamp.com/community/tutorials/finance-python-trading?utm_source=adwords_ppc&utm_medium=cpc&utm_campaignid=12492439679&utm_adgroupid=122563407721&utm_device=c&utm_keyword=python%20finance&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=504158802919&utm_targetid=aud-299261629654:dsa-301111479313&utm_loc_interest_ms=&utm_loc_physical_ms=9003205&gclid=CjwKCAiAm7OMBhAQEiwArvGi3M9vJV0Yyy8GZnMPcZeOKbhhEpTT3IkOKsl-sIL6Ug1rbFWreb2U2xoCS2MQAvD_BwE

- create two separate Simple Moving Averages (SMA) of a time series with differing lookback periods,
    let’s say, 40 days and 100 days. 
- long moving tends toward short moving.
- If the short moving average exceeds the long moving average then you go long.
- if the long moving average exceeds the short moving average then you exit.
'''
def mac(asset_df,price_col='Close', short_window=40,long_window=100, plot_signal=False):
    signals = pd.DataFrame(index=asset_df.index)
    signals['signal'] = 0.0 #initialization
    signals['short_mavg'] = asset_df[price_col].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = asset_df[price_col].rolling(window=long_window, min_periods=1, center=False).mean()
    # Create signals: only for the period greater than the shortest moving average window(start at short_window+1.th row)
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)   
    # Generate trading orders (-1sell,0hold,1buy)
    signals['positions'] = signals['signal'].diff()
    if plot_signal:
        fig = plt.figure()
        ax1 = fig.add_subplot(111,  ylabel='Price in $')
        asset_df[price_col].plot(ax=ax1, color='r', lw=2.)
        signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
        ax1.plot(signals.loc[signals.positions == 1.0].index, 
                signals.short_mavg[signals.positions == 1.0],
                '^', markersize=10, color='m')
        ax1.plot(signals.loc[signals.positions == -1.0].index, 
                signals.short_mavg[signals.positions == -1.0],
                'v', markersize=10, color='k')
        plt.show()
    return signals
'''
if cointegration resid too high(positively), sell the portfolio.
if cointegration resid too low(negatively), buy the portfolio
Use a threshold[0](multiple of volatility) to define the selling/buying signal.
Use a threshold[1](multiple of volatility) to define the clearing signal.
'''
def coint_strategy(coin_resid ,threshold=[1,2], plot_signal=False):
    signals = pd.DataFrame(index=coin_resid.index)
    signals['positions'] = 0.0 #initialization with all numb
    volatility_resid = coin_resid.std()
    #set up numb range to do nothing with code False
    active_loc =  (threshold[0]*volatility_resid > np.abs(coin_resid)) | (np.abs(coin_resid)> threshold[1]*volatility_resid)
    active_loc = (active_loc.values.flatten())
    #in sig_active: -1 for sell the portfolio,0 for clear ,1 for buy the portfolio
    sig_active = np.where(np.abs(coin_resid[active_loc])> threshold[1]*volatility_resid ,
                                            np.sign(coin_resid[active_loc])*(-1.0), 0.0)
    sig_active = pd.DataFrame({'index' :range(len(sig_active.flatten())),
                                'sig_active':sig_active.flatten()}).set_index('index')
    #in active positions: -1 for sell order, 0 for do nothing, 1 for buy order
    pos_active = np.sign(sig_active.diff()).fillna(sig_active)['sig_active'].values
    signals['positions'].loc[active_loc] = pos_active

    if plot_signal:
        fig = plt.figure()
        ax1 = fig.add_subplot(111,  ylabel='Coint residuals')
        coin_resid.plot(ax=ax1, lw=2.)
        ax1.plot(signals.loc[signals.positions == 1.0].index, 
                coin_resid[signals.positions == 1.0],
                '^', markersize=10, color='m')
        ax1.plot(signals.loc[signals.positions == -1.0].index, 
                coin_resid[signals.positions == -1.0],
                'v', markersize=10, color='k')
        plt.show()
    return signals
    


##########################test
if __name__ == "__main__":
    btc = pd.read_csv('./codetest/btc.csv')
    result = mac(asset_df=btc.set_index('timestamp'),
                price_col='Close', short_window=40,long_window=100, plot_signal=False)
    print(result)
    resid = pd.read_csv('./codetest/coin_resid.csv').set_index('timestamp')
    signals = coint_strategy(coin_resid= resid, threshold=[0.5,1], plot_signal=False)
    print(signals[signals['positions']==1])




    


