'''
List of utility content:
- log_return
- mysplit
- WindowGenerator
- compile_and_fit
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

############################################################################log return
def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)
############################################################################split data
def mysplit(df, **split_para):
    df = df[['Count','Open','High','Low','Close','Volume','VWAP','Target']]
    train_df,val_df = train_test_split(df,train_size=split_para['train_size'],
                                                test_size=split_para['val_size'],shuffle=False)
    #holdout/test sets
    n = len(df) #rows
    test_start = int(n*(1-split_para['train_size']-split_para['val_size']))
    test_df = df[test_start:]

    return train_df, val_df, test_df

############################################################################window class
class WindowGenerator():
  def __init__(self, input_width, label_width, shift, train_df, val_df, test_df,
               label_columns=None,**kwargs):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window width.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift #offset size
    self.total_window_size = input_width + shift

    #input/label index and slice
    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    ##display output
    return '\n'.join([
        f'Input width={self.input_width},Label width={self.label_width},Offset width={self.shift}',
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False,
      batch_size=32,)
  ds = ds.map(self.split_window)
  return ds

WindowGenerator.split_window = split_window
WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  result = getattr(self, '_example', None)
  if result is None:
    result = next(iter(self.train))
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

def plot(self, model=None, plot_col='Close', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [m]')

WindowGenerator.plot = plot
##############################################################################compile and fit
def compile_and_fit(model, window, **compilefitpara):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=2,
                                                      mode='min',restore_best_weights=False)
  if 'pretrained' not in compilefitpara:
    print('Compile...')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError(),'mse'])
    print('Train...')
    history = model.fit(window.train, epochs=compilefitpara['MAX_EPOCHS'],
                        validation_data=window.val,
                        verbose = compilefitpara['verbose'],
                        callbacks=[early_stopping])
  elif compilefitpara['pretrained']:
    # designed for updating trained model with new data
    print('Updating pretrained model...')
    history = model.fit(window.train, epochs=compilefitpara['MAX_EPOCHS'],
                            validation_data=window.val,
                            verbose = 0,
                            callbacks=[early_stopping])
  return history

