'''
A general process to take care of blocked k-fold process for time series model.
'''
if __name__ != '__main__':
    from script.RNN.kfoldwindow import BlockingTimeSeriesSplit
    from script.RNN.myModel import mymodel, myfittedmodel

import tensorflow as tf
import numpy as np
from sklearn import metrics
import itertools

def blockkfoldcv(df,K,model,withholdout:bool=True,holdoutpct=0.1):
    '''
    return the k-fold metrics on the model
    '''
    if withholdout:
        #holdout/test sets
        n = len(df)
        test_df = df[int(n*(1-holdoutpct)):]
        tscv = BlockingTimeSeriesSplit(n_splits = K,X=df[:int(n*(1-holdoutpct))])
    else:
        tscv = BlockingTimeSeriesSplit(n_splits = K,X=df)
    fold = 0
    oos_y = []
    oos_pred = []
    fold_score=[]
    #k-fold loop
    for train_index, val_index in tscv.split():
        fold+=1
        print(f"Fold #{fold}")
        #train, validation sets in the fold
        cv_train, cv_val = df.iloc[train_index], df.iloc[val_index]
        num_features = df.shape[1]
        #normalization
        train_mean = cv_train.mean()
        train_std = cv_train.std()
        cv_train = (cv_train - train_mean) / train_std #scipy.stats.zscore(train_df,ddof=1)
        cv_val = (cv_val - train_mean) / train_std
        if withholdout:
            test_df = (test_df - train_mean) / train_std
        #fit model
        wide_window,lstm_model = myfittedmodel(model=model,
                                                input_width=30, label_width=30, shift=1,
                                                train_df=cv_train, val_df=cv_val, test_df=test_df,
                                                label_columns=['Close'],fit=True)

        # Measure MSE error for this fold
        score = lstm_model.evaluate(wide_window.val, verbose=0,return_dict=True)
        fold_score.append(score['mse'])
        print(f"Fold score (MSE): {score['mse']}")

        # prediction on validation set
        pred = []
        label = []
        for input_batch,label_batch in wide_window.val:
            pred_batch = lstm_model(input_batch)
            pred.append(list(pred_batch.numpy().flatten()))
            label.append(list(label_batch.numpy().flatten()))
        #flatten list of lists
        pred = np.array(list(itertools.chain(*pred)))
        label = np.array(list(itertools.chain(*label)))
        oos_y.append(label)
        oos_pred.append(pred) 
    # Build the oos prediction list and calculate the error.
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    oos_score = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))
    print(f"Final, out of sample score (RMSE): {oos_score}")  
    return  oos_score
