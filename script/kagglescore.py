import numpy as np
import pandas as pd

def weighted_correlation(a, b, weights):
    '''
    'a' and 'b' are the expected and predicted targets, 
        and ' weights' include the weight of each row, determined by its asset
    '''
    w = np.ravel(weights)
    a = np.ravel(a)
    b = np.ravel(b)

    sum_w = np.sum(w)
    mean_a = np.sum(a * w) / sum_w
    mean_b = np.sum(b * w) / sum_w
    var_a = np.sum(w * np.square(a - mean_a)) / sum_w
    var_b = np.sum(w * np.square(b - mean_b)) / sum_w

    cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
    corr = cov / np.sqrt(var_a * var_b)

    return corr

if __name__ == '__main__':
    sup_train = pd.read_csv('./data/supplemental_train.csv')[['timestamp','Asset_ID','Target']]
    weight = pd.read_csv('./data/asset_details.csv')
    submission = pd.read_csv('./data/kagglesubmission/submission.csv')#row_id
    test = []
    for i in range(4):
        test_csv= pd.read_csv(f'./data/kagglesubmission/{i}test_df.csv')
        test.append(test_csv)
    pred = pd.concat(test).merge(submission, on = 'row_id')[['timestamp','Asset_ID','row_id','Target']]
    score_df =pred.merge(sup_train, how='left', on=['timestamp','Asset_ID'],suffixes = ('_pred','_true'))
    score_df = score_df.merge(weight,how='left',on='Asset_ID')
    print(score_df)
    s = weighted_correlation(a= score_df['Target_pred'], b= score_df['Target_true'], weights=score_df['Weight'])
    print(s)