import numpy  as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class BlockingTimeSeriesSplit():
    '''
    1.Blockingbased sklearn.model_selection.TimeSeriesSplit
    2.offer train_df and val_df into WindowGenerator
    '''
    def __init__(self, n_splits, X):
        self.n_splits = n_splits
        self.X = X
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X=None, y=None, groups=None):
        x_none = getattr(self, 'X', None)
        if x_none is not None:
            n_samples = len(self.X)
        else:
            #in case of TimeSeriesSplit
            n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            #return a generator
            train_ind, test_ind = indices[start: mid], indices[mid + margin: stop]
            yield train_ind, test_ind



########################################visualize the split
def plot_cv_indices(cv, X,y=None, lw=10):
    """
    Create a sample plot for indices of a cross-validation object.
    https://goldinlocks.github.io/Time-Series-Cross-Validation/
    """
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    fig, ax = plt.subplots(figsize=(10, 5))
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                c=indices, marker='_', lw=lw, cmap=cmap_cv,
                vmin=-.2, vmax=1.2)

    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
            c=y, marker='_', lw=lw, cmap=cmap_data)
    ax.set_title(f'{type(cv).__name__}', fontsize=15)
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
        ['Testing set', 'Training set'], loc=(1.02, .8))
    plt.tight_layout()
    fig.subplots_adjust(right=.7)
    plt.show()

####################################test
if __name__ =='__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    csv_path = '/Users/dingxian/Documents/GitHub/Crypto_Forecasting_kaggle/codetest/btc.csv'
    df = pd.read_csv(csv_path)
    df = df[-70000:]
    date_time = pd.to_datetime(df.pop('timestamp'),unit='s')
    df = df[['Count','Open','High','Low','Close','Volume','VWAP']]
    train_df,val_df = train_test_split(df,train_size=0.70,test_size=0.20,shuffle=False)
    n = len(df) #rows
    test_df = df[int(n*0.9):]
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std 
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    tscv = BlockingTimeSeriesSplit(X = train_df,n_splits = 5)
    for train_index, test_index in tscv.split():
        cv_train, cv_test = train_df.iloc[train_index], train_df.iloc[test_index]
        print(train_index, test_index)
    plot_cv_indices(cv=tscv,X=tscv.X)
    