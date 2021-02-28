import os
# ==========================================================
# Constants
# ==========================================================
DROPOUT = True
LSTM    = 1
GRU     = 2
modes_dict  = {LSTM: 'Long_Short_Term_memory',
               GRU:  'Gated_Recurrent_Unit'}  # modes dictionary
# ===========================================================
# FLow Control
# ===========================================================
FIND_OPTIMAL_LEARNING_RATE = False  # change to perform the learning rate test
PER_MODE_CONVERGENCE       = False  # change to perform the convergence training with MU and EPOCH_NUM for every mode
TRAIN_SPECIFIC_MODE        = True  # change to perform training for one mode
LOAD_AND_TEST              = False  # change if you want to load a pre-trained model, and perform accuracy test

# ==========================================================
# important paths
# ==========================================================
path_logs     = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Ex2\\RNN\\data\\logs'
path_models   = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Ex2\\RNN\\data\\models'
path_datasets = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Ex2\\RNN\\data\\datasets'

path_train_data = os.path.join(path_datasets, "ptb.train.txt")
path_val_data   = os.path.join(path_datasets, "ptb.valid.txt")
# ============================================================
# Global variables of the net
# ============================================================
# ----------------------
# Hyper parameters
# ----------------------
EPOCH_NUM       = 5
MU              = 1e-2  # learning rate
BATCH_SIZE      = 20
HIDDEN          = 200   # number of neurons in the hidden layer
H_LAYER_NUM     = 2     # number of hidden layers
SEQUENCE_LENGTH = 20
EMBED_SIZE      = 1000
DROPOUT_RATE    = 0.5
GRAD_CLIP       = 5
SCHEDULER_STEP  = 4
SCHEDULER_GAMMA = 0.5
