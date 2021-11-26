import tensorflow as tf

DEVICE = "TPU" #or "GPU"
SEED = 42
EPOCHS = 20
DEBUG = False
N_ASSETS = 14 #14 assets
WINDOW_SIZE = 15 #15mins
BATCH_SIZE = 1024
PCT_VALIDATION = 5 # last 5% of the data are used as validation set
if __name__ != '__main__':
    print(f'N_ASSETS = {N_ASSETS}, WINDOW_SIZE ={WINDOW_SIZE}, BATCH_SIZE = {BATCH_SIZE},EPOCHS = {EPOCHS},PCT_VALIDATION={PCT_VALIDATION}')
 
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None
    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except: print("failed to initialize TPU")
    else: DEVICE = "GPU"

if DEVICE != "TPU": strategy = tf.distribute.get_strategy()
if DEVICE == "GPU": print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync