'''
The evaluation metric is weighted correlation as opposed to a weighted mean of correlation.

The metric is defined as follows, where 'a', 'b' and 'weights' are vectors of the same length.

'a' and 'b' are the expected and predicted targets, and ' weights' include the weight of each row, 
            determined by its asset.
'''

import numpy as np

def weighted_correlation(a, b, weights):

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