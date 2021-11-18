
from script.RNN.myModel import modelcandidate
from ann_visualizer.visualize import ann_viz
from keras.utils.vis_utils import plot_model
import pandas as pd
from sklearn.model_selection import train_test_split
##data and date
csv_path = './codetest/btc.csv'
df = pd.read_csv(csv_path)
df = df[-70000:]
date_time = pd.to_datetime(df.pop('timestamp'),unit='s')
df = df[['Count','Open','High','Low','Close','Volume','VWAP']]

#train, validation sets
train_df,val_df = train_test_split(df,train_size=0.70,test_size=0.20,shuffle=False)
n = len(df) #rows
test_df = df[int(n*0.9):]
num_features = df.shape[1]
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std #scipy.stats.zscore(train_df,ddof=1)
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#model and result
wide_window,lstm_model = modelcandidate(input_width=30, label_width=30, shift=1,
                                                train_df=train_df, val_df=val_df, test_df=test_df,
                                                label_columns=['Close'])

plot_model(lstm_model, to_file='./pic/model_plot.png', show_shapes=True, show_layer_names=True)