from tensorflow import keras
import numpy as np


class sample_generator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, length):
        #(train, targets) arrays tuple
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.length = length#window size
        self.size = len(x_set)#unique timestamp size
    def __len__(self): 
        #number of batchs
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    def __getitem__(self, idx):
        #(train, targets) tuple
        batch_x=[]
        batch_y=[]
        for i in range(self.batch_size):
            start_ind = self.batch_size*idx + i
            end_ind = start_ind + self.length 
            if end_ind <= self.size:
                batch_x.append(self.x[start_ind : end_ind])
                batch_y.append(self.y[end_ind -1])
        return np.array(batch_x), np.array(batch_y)

