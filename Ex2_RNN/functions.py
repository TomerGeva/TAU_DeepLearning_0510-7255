from Config import *
import torch
# import torchvision
import os
import numpy as np
# import pandas as pd
# import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader


# **********************************************************************************************************************
# THIS SECTION DEALS WITH FUNCTIONS REGARDING WITH THE DATABASES
# **********************************************************************************************************************
def import_data_sets():
    """
    :return: this function returns the train database and validation databases
    """
    # -------------------------------------------------------------------
    # downloading the relevant datasets
    # -------------------------------------------------------------------
    train_data_path = os.path.join(path_datasets, "ptb.train.txt")
    with open(train_data_path, 'r') as train_file:
        train_data = train_file.read()

    val_data_path = os.path.join(path_datasets, "ptb.valid.txt")
    with open(val_data_path, 'r') as val_file:
        val_data = val_file.read()

    return train_data, val_data


def encode_text(data):
    """
    :param data: a dataset listed as sentences
    :return: The function returns the vocabulary used and the encoded text, using this
             vocabulary
    """
    # --------------------------------------------------------------------------------------
    # Set creates an un-indexed and unorganized unique values of the words in the text.
    # This set is then turned into a list
    # --------------------------------------------------------------------------------------
    vocab = list(set(data.split(' ')))
    # --------------------------------------------------------------------------------------
    # Removing unwanted words
    # --------------------------------------------------------------------------------------
    vocab = remove_unnecessary_words(vocab)
    # --------------------------------------------------------------------------------------
    # Creating a dictionary with the numbering of the vocabulary
    # --------------------------------------------------------------------------------------
    int2word = dict(enumerate(vocab))  # maps integers to words
    # --------------------------------------------------------------------------------------
    # At this point we have a dictionary where the keys are integers, and the values are
    # the words. We need the opposite, and therefore, we weill create a new dictionary
    # --------------------------------------------------------------------------------------
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # dict.items() creates a tuple pairs of (key, value)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    word2int = {word: num for num, word in int2word.items()}  # maps words to unique integers
    # --------------------------------------------------------------------------------------
    # Encoding the text words into the encoded text
    # --------------------------------------------------------------------------------------
    encoded = np.array([word2int[word] for word in data.split(' ') if word in word2int])
    return int2word, encoded


def remove_unnecessary_words(data, unnecessary=["<unk>", "N", "$", ""]):
    """
    :param data: a dataset listed as sentences
    :param unnecessary: a list of unwanted words
    :return: the function returns a list without the unwanted words
    """
    return [word for word in data if word not in unnecessary]


def save_state_train(trainer, filename, net, epoch, lr, loss):
    """Saving model and optimizer to drive, as well as current epoch and loss
    # When saving a general checkpoint, to be used for either inference or resuming training, you must save more
    # than just the model’s state_dict.
    # It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are
    # updated as the model trains.
    """
    path = os.path.join(path_models, filename)
    data_to_save = {'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': loss,
                    'lr': lr
                    }
    torch.save(data_to_save, path)


# **********************************************************************************************************************
# THIS SECTION DEALS WITH FUNCTIONS REGARDING TENSOR MANIPULATION, BOTH ARE NOT NEEDED!!!!!!!!!!!!!!
# **********************************************************************************************************************
def int2onehot(sequence_vec, length):
    """
    :param sequence_vec: a vector with dimensions of SEQUENCE_LENGTH
    :param length: the length of the 1-hot vector, i.e. the vocabulary size
    :return: the function returns a 3D torch matrix, where the dimensions are: SEQUENCE_LENGTH X length
    """
    if torch.is_tensor(sequence_vec):
        sequence_vec = sequence_vec.data.numpy()

    eye_tensor = torch.eye(length)
    mat = None
    for loc in sequence_vec:
        if mat is None:
            # mat = np.expand_dims(eye_vec[loc], axis=0)
            mat = eye_tensor[loc].unsqueeze(dim=0)
        else:
            # mat = np.concatenate((mat, np.expand_dims(eye_vec[loc], axis=0)), axis=0)
            mat = torch.cat((mat, eye_tensor[loc].unsqueeze(dim=0)), dim=0)
    return mat


def int2onehot_fast(sequence_vec, length):
    """
    :param sequence_vec: a vector with dimensions of SEQUENCE_LENGTH
    :param length: the length of the 1-hot vector, i.e. the vocabulary size
    :return: the function returns a 3D torch matrix, where the dimensions are: SEQUENCE_LENGTH X length
    """
    mat = np.zeros((len(sequence_vec), length))
    ii = [ii for ii in range(len(sequence_vec))]
    mat[ii, sequence_vec] = 1
    return torch.from_numpy(mat)

