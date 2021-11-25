'''
windowing + model.compile+model.fit
'''
if __name__ != '__main__':
    #importing from the global environment at CRYPTO_FORCASTING_KAGGLE
    from script.RNN.window import WindowGenerator
    from script.RNN.compilefit import compile_and_fit

import tensorflow as tf

def mymodel():
    '''
    design the base lstm models for compile and fit
    '''
    model_list = []
    ## model1 single time-step
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units = 20, return_sequences=True,
                            activation='tanh', recurrent_activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=1))
    model_list.append(model)
    ## model2 multiple time-step
    
    return model_list

def myfittedmodel(model,**windowpara):
    # windowing
    wide_window = WindowGenerator(**windowpara)
    # fit the model
    if 'type' not in windowpara:
        history = compile_and_fit(model = model, window = wide_window, 
                                    MAX_EPOCHS = 2,verbose=0)
    elif windowpara['type'] =='update':
        history = compile_and_fit(model = model, window = wide_window, 
                                    MAX_EPOCHS = 2,verbose=0, pretrained=True)
    return wide_window,model

