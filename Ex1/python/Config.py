# ==========================================================
# Constants
# ==========================================================
NO_REGULARIZATION   = 0
DROP_OUT            = 1
WEIGHT_DECAY        = 2
BATCH_NORMALIZATION = 3
X_SIZE              = 28
Y_SIZE              = 28

# ===========================================================
# FLow Control
# ===========================================================
FIND_OPTIMAL_LEARNING_RATE = False  # change to perform the learning rate test
PER_MODE_CONVERGENCE       = False  # change to perform the convergence training with MU and EPOCH_NUM for every mode
TRAIN_SPECIFIC_MODE        = True  # change to perform training for one mode
LOAD_AND_TEST              = False  # change if you want to load a pre-trained model, and perform accuracy test
# ===========================================================
# dictionaries
# ===========================================================
labels_dict = {0: 'T-Shirt',
               1: 'Trouser',
               2: 'Pullover',
               3: 'Dress',
               4: 'Coat',
               5: 'Sandal',
               6: 'Shirt',
               7: 'Sneaker',
               8: 'Bag',
               9: 'Ankle Boot'}  # label dictionary
modes_dict  = {BATCH_NORMALIZATION: 'batch_normalization',
               DROP_OUT:            'drop_out',
               WEIGHT_DECAY:        'weight_decay',
               NO_REGULARIZATION:   'no_regularization'}  # modes dictionary

# ==========================================================
# important paths
# ==========================================================
path_logs   = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Ex1\\data\\logs'
path_models = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Ex1\\data\\models'

# ============================================================
# Global variables of the net
# ============================================================
# ----------------------
# Hyper parameters
# ----------------------
EPOCH_NUM        = 1
MU               = 1e-2  # learning rate
BATCH_SIZE       = 15
DROPOUT_RATE     = 0.5
WEIGHTDECAY_RATE = 0.01

# ----------------------
# Network topology
# ----------------------
layer_description = {0: 'conv',
                     1: 'ReLU',
                     2: 'pool',
                     3: 'conv',
                     4: 'ReLU',
                     5: 'pool',
                     6: 'linear'}
# ++++++++++++++++++++++
# conv layers topology
# ++++++++++++++++++++++
# Number of filters in each filter layer
filter_num   = [16,  # first conv
                32]  # second conv
# Filter sizes for each filter layer
kernel_sizes = [5,  # first layer
                5]  # second layer
# Stride values of the convolution layers
strides      = [1,  # first layer
                1]  # second layer
# Padding values of the convolution layers
padding      = [2,  # first conv
                2]  # second conv
# Max pool size
max_pool_size   = 2
# ++++++++++++++++++++++
# FC layer topology
# ++++++++++++++++++++++
fc_layers = [84,  # first FC
             10]  # output FC
