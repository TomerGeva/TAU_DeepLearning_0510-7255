import torchvision.transforms as transforms
# ============================================================
# Constants
# ============================================================
DCGAN  = 0
WGAN   = 1
X_SIZE = 64
Y_SIZE = 64
Z_DIM  = 100
TRANSFORM_NORM   = 0.5
INIT_WEIGHT_MEAN = 0
INIT_WEIGHT_STD  = 0.02

# ============================================================
# FLow Control
# ============================================================
FIND_OPTIMAL_LEARNING_RATE = False  # change to perform the learning rate test
PER_MODE_CONVERGENCE       = False  # change to perform the convergence training with MU and EPOCH_NUM for every mode
TRAIN_SPECIFIC_MODE        = False   # change to perform training for one mode
LOAD_AND_TEST              = True  # change if you want to load a pre-trained model, and perform accuracy test
# ============================================================
# dictionaries
# ============================================================
LABELS_DICT = {0: 'T-Shirt',
               1: 'Trouser',
               2: 'Pullover',
               3: 'Dress',
               4: 'Coat',
               5: 'Sandal',
               6: 'Shirt',
               7: 'Sneaker',
               8: 'Bag',
               9: 'Ankle Boot'}  # label dictionary
MODES_DICT  = {DCGAN: 'DCGAN',
               WGAN:   'WGAN',
               }  # modes dictionary

# ============================================================
# important paths
# ============================================================
PATH_LOGS   = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Ex3\\data\\logs'
PATH_MODELS = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Ex3\\data\\models'

# ============================================================
# Global variables of the net
# ============================================================
# --------------------------------------------------------
# Hyper parameters
# --------------------------------------------------------
EPOCH_NUM        = 1
MU               = 2e-4  # learning rate
ALPHA            = 0.2   # leaky ReLU parameter for discriminator
BATCH_SIZE       = 128
IMG_CHANNELS     = 1

# --------------------------------------------------------
# Network topology
# --------------------------------------------------------
# Number of filters in each filter layer
"""
DISC_FEATURE_NUMS = [128,   # first conv
                     256,   # second conv
                     512,   # third conv
                     1024]  # fourth conv
GEN_FEATURE_NUMS  = [1024,  # first conv
                     512,   # second conv
                     256,   # third conv
                     128]   # fourth conv
"""
DISC_FEATURE_NUMS = [128,   # first conv
                     256,   # second conv
                     512,   # third conv
                     1024]  # fourth conv
GEN_FEATURE_NUMS  = [1024,  # first conv
                     512,   # second conv
                     256,   # third conv
                     128]   # fourth conv
# Filter sizes for each filter layer
KERNEL_LIST  = [4,  # first layer
                4,  # second layer
                4,  # third layer
                4,  # fourth layer
                4]  # fifth layer
# Stride values of the convolution layers
DISC_STRIDE_LIST = [2,  # first layer
                    2,  # second layer
                    2,  # third layer
                    2,  # fourth layer
                    1]  # fifth layer
GEN_STRIDE_LIST  = [1,  # first layer
                    2,  # second layer
                    2,  # third layer
                    2,  # fourth layer
                    2]  # fifth layer
# Padding values of the convolution layers
DISC_PADDING_LIST = [1,  # first layer
                     1,  # second layer
                     1,  # third layer
                     1,  # fourth layer
                     0]  # fifth layer
GEN_PADDING_LIST  = [0,  # first layer
                     1,  # second layer
                     1,  # third layer
                     1,  # fourth layer
                     1]  # fifth layer
