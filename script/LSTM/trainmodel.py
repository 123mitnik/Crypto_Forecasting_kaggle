import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow import keras

########################custom metrics
def MaxCorrelation(y_true,y_pred): 
    return -tf.math.abs(tfp.stats.correlation(y_pred,y_true, sample_axis=None, event_axis=None))
def Correlation(y_true,y_pred): 
    return tf.math.abs(tfp.stats.correlation(y_pred,y_true, sample_axis=None, event_axis=None))
##########################custom loss func
def masked_mse(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0.)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.mean_squared_error(y_true = y_true_masked, y_pred = y_pred_masked)

def masked_mae(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0.)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.mean_absolute_error(y_true = y_true_masked, y_pred = y_pred_masked)

def masked_cosine(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0.)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.cosine_similarity(y_true_masked, y_pred_masked)
############################define lstm layer
def get_squence_model(x):
    x = layers.LSTM(units=32, return_sequences=True)(x)
    return x

############################compile model
'''
- Lambda layer needed for assets separation;
- Masking layer. Generated records (filled gaps) has zeros as features values, so they are not used in the computations.
- Our sequence model architecture(lstm)
- Concatenation layer
- Dense layer
'''
def get_model(n_assets = 14, trainshape=(15,14,12)):
    #Keras tensor
    x_input = keras.Input(shape=trainshape)
    ## parallel sequence model branches
    branch_outputs = []
    for i in range(n_assets):
        # Slicing the ith asset: x_input into x of the lambda function
        a = layers.Lambda(lambda x: x[:,:, i])(x_input) #lambda layer
        a = layers.Masking(mask_value = 0., )(a) #masking layer
        #a = get_squence_model(a)#lstm layer
        a = layers.LSTM(units=32, return_sequences=True)(a)
        a = layers.GlobalAvgPool1D()(a)#globalavgpool layer
        branch_outputs.append(a)
    
    x = layers.Concatenate()(branch_outputs)
    x = layers.Dense(units = 128)(x)#dense layer 0
    out = layers.Dense(units = n_assets)(x)#dense layer 1
    model = keras.Model(inputs=x_input, outputs=out)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
                    loss = masked_cosine, metrics=[Correlation])
    return model


##########test
