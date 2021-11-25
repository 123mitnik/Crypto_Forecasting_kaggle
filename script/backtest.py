'''
testing the strategy on relevant historical data to make sure that it’s an actual viable strategy.
- A data handler, which is an interface to a set of data,
- A strategy, which generates a signal to go long or go short based on the data,
- A portfolio, which generates orders and manages Profit & Loss (also known as “PnL”), and
- An execution handler, which sends the order to the broker and receives the “fills” or signals that the stock has been bought or sold.
'''
import pandas as pd
import matplotlib.pyplot as plt

def bt(asset_df,signals,strategy:str ,initial_capital = float(100000.0), price_col = 'Close', 
        shares_signal=100, plot_balance=False):
    # Create a DataFrame `positions`:shares of stock
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    list_valid_strategy=['mac','coint']
    if strategy not in list_valid_strategy:
        raise ValueError(f"Wrong input: strategy should be a string as: {list_valid_strategy} ")

    if strategy == 'mac':
        positions['Shares'] = shares_signal*signals['signal'] #shares_signal * (0,1)
        # Store the difference in shares owned= shares_signal*signals['positions']
        pos_diff = positions.diff() #(-shares_signal,0,+shares_signal)
        # Initialize the portfolio() as a dataframe
        portfolio = positions
        # `holdings` stores the value of the positions, 
        #  sum(axis=1) is summation over (row)all the stocks' position
        portfolio['holdings'] = (positions.multiply(asset_df[price_col], axis=0)).sum(axis=1)
        # `cash` the capital that you still have left to spend
        portfolio['cash'] = initial_capital - (pos_diff.multiply(asset_df[price_col], axis=0)).sum(axis=1).cumsum()   
    elif strategy == 'coint':
        ##asset_df is the coint resid which should be a series not a one-column df.
        pos_diff = shares_signal*signals #shares_signal * (-1,0,1)
        positions['Shares'] = pos_diff.cumsum()
        portfolio = positions
        portfolio['holdings'] = (positions.multiply(asset_df, axis=0)).sum(axis=1)
        portfolio['cash'] = initial_capital - (pos_diff.multiply(asset_df, axis=0)).sum(axis=1).cumsum()
        
    # `balance` the sum of your cash and the holdings
    portfolio['balance'] = portfolio['cash'] + portfolio['holdings']
    # `returns` pct_return(daily or frequency-ly)
    portfolio['returns'] = portfolio['balance'].pct_change()
    if plot_balance:
        # Create a figure
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Balance in $')
        # Plot the equity curve in dollars
        portfolio['balance'].plot(ax=ax1, lw=2.)

        ax1.plot(portfolio.loc[signals.positions == 1.0].index, 
                portfolio.balance[signals.positions == 1.0],
                '^', markersize=10, color='m')
        ax1.plot(portfolio.loc[signals.positions == -1.0].index, 
                portfolio.balance[signals.positions == -1.0],
                'v', markersize=10, color='k')
        plt.show()
    return portfolio

if __name__ == "__main__":
    from strategy import mac, coint_strategy
    btc = pd.read_csv('./codetest/btc.csv')
    sig = mac(asset_df=btc.set_index('timestamp'),
                price_col='Close', short_window=40,long_window=100, plot_signal=False)
    result = bt(asset_df=btc.set_index('timestamp'), signals=sig, strategy='mac',
                initial_capital = float(100000.0), price_col= 'Close', shares_signal=100, plot_balance=False)
    
    resid = pd.read_csv('./codetest/coin_resid.csv', index_col=0,squeeze=True)
    sig = coint_strategy(coin_resid= resid, threshold=[0.5,1], plot_signal=False)
    result = bt(asset_df=resid, signals=sig, strategy='coint',
                initial_capital = float(100000.0), shares_signal=100, plot_balance=False)
    print(result)
  










