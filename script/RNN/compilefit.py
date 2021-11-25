import tensorflow as tf
from datetime import datetime

def compile_and_fit(model, window, **compilefitpara):
  '''
  model could be a pretrained or new 
  '''
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=2,
                                                      mode='min',restore_best_weights=False)
  #Configures the model for training.
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
    print('Updating pretrained model...')
    history = model.fit(window.train, epochs=compilefitpara['MAX_EPOCHS'],
                            validation_data=window.val,
                            verbose = 0,
                            callbacks=[early_stopping])
  return history