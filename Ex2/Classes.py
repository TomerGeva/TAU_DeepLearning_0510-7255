from Config import *
from functions import *
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split


# **********************************************************************************************************************
# THIS SECTION DEALS WITH CLASSES REGARDING WITH THE DATABASE, AND CREATION OF THE DATALOADERS
# **********************************************************************************************************************
# ============================================================
# defining word dataset class, containing
# ============================================================
class WordDataset:
    """
    This class is used to store the word database, as well as the codexes
    """
    def __init__(self):
        """
        :param csv_file: Path to the wanted database location
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
        self.data = " ".join([train_data, val_data])

        # --------------------------------------------------------------------------------------
        # Set creates an un-indexed and unorganized unique values of the words in the text.
        # This set is then turned into a list
        # --------------------------------------------------------------------------------------
        self.vocab = list(set(self.data.split(' ')))

        # --------------------------------------------------------------------------------------
        # Removing unwanted words
        # --------------------------------------------------------------------------------------
        self.vocab = remove_unnecessary_words(self.vocab)

        # --------------------------------------------------------------------------------------
        # Creating a dictionary with the numbering of the vocabulary
        # --------------------------------------------------------------------------------------
        self.int2word = dict(enumerate(self.vocab))

        # --------------------------------------------------------------------------------------
        # At this point we have a dictionary where the keys are integers, and the values are
        # the words. We need the opposite, and therefore, we weill create a new dictionary
        # --------------------------------------------------------------------------------------
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # dict.items() creates a tuple pairs of (key, value)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.word2int = {word: num for num, word in self.int2word.items()}

        # --------------------------------------------------------------------------------------
        # Encoding the text words into the encoded text
        # --------------------------------------------------------------------------------------
        self.encoded = np.array([self.word2int[word] for word in self.data.split(' ') if word in self.word2int])

    def __len__(self):
        # --------------------------------------------------------------------------------------
        # the length of the database is the number of sequences
        # --------------------------------------------------------------------------------------
        return len(self.vocab)


# ============================================================
# defining word sequence class, dealing with sequence popping
# ============================================================
class WordSequenceDatasetSampler(Dataset):
    """
    This class is used to store the sequences. the sequence length is set to be SEQUENCE_LENGTH
    the global constant
    """
    def __init__(self, csv_file, word_database, sequence_len=20, transform=None):
        """
        :param csv_file: Path to the wanted database location
        :param word_database: the complete word database!
        :param sequence_len: length of each sequence in the database
        :param transform: transformation if needed, in our case, this converts to tensor
        """
        self.transform = transform
        # --------------------------------------------------------------------------------------
        # Reading the data file from the path
        # --------------------------------------------------------------------------------------
        with open(csv_file, 'r') as file:
            self.csv_data = file.read()

        # --------------------------------------------------------------------------------------
        # Set creates an un-indexed and unorganized unique values of the words in the text.
        # This set is then turned into a list
        # --------------------------------------------------------------------------------------
        self.vocab = word_database.vocab

        # --------------------------------------------------------------------------------------
        # Creating a dictionary with the numbering of the vocabulary
        # --------------------------------------------------------------------------------------
        self.int2word = word_database.int2word

        # --------------------------------------------------------------------------------------
        # At this point we have a dictionary where the keys are integers, and the values are
        # the words. We need the opposite, and therefore, we weill create a new dictionary
        # --------------------------------------------------------------------------------------
        self.word2int = word_database.word2int

        # --------------------------------------------------------------------------------------
        # Encoding the text words into the encoded text
        # --------------------------------------------------------------------------------------
        encoded = np.array([self.word2int[word] for word in self.csv_data.split(' ') if word in self.word2int])

        # --------------------------------------------------------------------------------------
        # arranging the encoded data as a matrix where each row is a sequence
        # --------------------------------------------------------------------------------------
        n = sequence_len
        self.encoded_sequences     = [encoded[i:i+n] for i in range(0, len(encoded), n)]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # popping the last row, not with length of sequence length
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.encoded_sequences = self.encoded_sequences[:-1]

    def __len__(self):
        # --------------------------------------------------------------------------------------
        # the length of the database is the number of sequences
        # --------------------------------------------------------------------------------------
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        # --------------------------------------------------------------------------------------
        # popping a sequence from the database
        # --------------------------------------------------------------------------------------
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # --------------------------------------------------------------------------------------
        # Creating the output vector where the output of each word is always the next word.
        # Performing wrap-around where the output of the last word is the first word
        # --------------------------------------------------------------------------------------
        in_sequence  = self.encoded_sequences[idx]
        try:
            out_sequence = np.append(in_sequence[1:], self.encoded_sequences[idx+1][0])
        except IndexError:
            out_sequence = np.append(in_sequence[1:], self.encoded_sequences[0][0])

        # --------------------------------------------------------------------------------------
        # Converting into one-hot vectors
        # --------------------------------------------------------------------------------------
        in_sequence  = int2onehot(in_sequence, len(self.vocab))
        # out_sequence = int2onehot(out_sequence, len(self.vocab))

        # --------------------------------------------------------------------------------------
        # Creating a complete sample
        # --------------------------------------------------------------------------------------
        sample = {'input':  in_sequence,
                  'output': out_sequence}

        if self.transform:
            sample = self.transform(sample)

        return sample


class WordSequenceDatasetIterable(IterableDataset):
    def __init__(self, csv_file, word_database, sequence_len=20, transform=None):
        """
        :param csv_file: Path to the wanted database location
        :param word_database: the complete word database!
        :param sequence_len: length of each sequence in the database
        :param transform: transformation if needed, in our case, this converts to tensor
        """
        self.transform = transform
        # --------------------------------------------------------------------------------------
        # Reading the data file from the path
        # --------------------------------------------------------------------------------------
        with open(csv_file, 'r') as file:
            self.csv_data = file.read()

        # --------------------------------------------------------------------------------------
        # Set creates an un-indexed and unorganized unique values of the words in the text.
        # This set is then turned into a list
        # --------------------------------------------------------------------------------------
        self.vocab = word_database.vocab

        # --------------------------------------------------------------------------------------
        # Creating a dictionary with the numbering of the vocabulary
        # --------------------------------------------------------------------------------------
        self.int2word = word_database.int2word

        # --------------------------------------------------------------------------------------
        # At this point we have a dictionary where the keys are integers, and the values are
        # the words. We need the opposite, and therefore, we weill create a new dictionary
        # --------------------------------------------------------------------------------------
        self.word2int = word_database.word2int

        # --------------------------------------------------------------------------------------
        # Encoding the text words into the encoded text
        # --------------------------------------------------------------------------------------
        encoded = np.array([self.word2int[word] for word in self.csv_data.split(' ') if word in self.word2int])

        # --------------------------------------------------------------------------------------
        # arranging the encoded data as a matrix where each row is a sequence
        # --------------------------------------------------------------------------------------
        n = sequence_len
        self.encoded_sequences     = [encoded[i:i+n] for i in range(0, len(encoded), n)]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # popping the last row, not with length of sequence length
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.encoded_sequences = self.encoded_sequences[:-1]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # creating the output sequences
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        out_sequences = []
        for ii, in_sequence in enumerate(self.encoded_sequences):
            try:
                out_sequence = np.append(in_sequence[1:], self.encoded_sequences[ii + 1][0])
            except IndexError:
                out_sequence = np.append(in_sequence[1:], self.encoded_sequences[0][0])
            out_sequences.append(out_sequence)
        self.encoded_outputs = out_sequences

    def __len__(self):
        # --------------------------------------------------------------------------------------
        # the length of the database is the number of sequences
        # --------------------------------------------------------------------------------------
        return len(self.encoded_sequences)

    def __iter__(self):
        return iter(tuple(zip(self.encoded_sequences, self.encoded_outputs)))


# ============================================================
# defining transform class, converting 1D arrays to tensors
# ============================================================
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        in_sequence, out_sequence = sample['input'], sample['output']

        # in_sequence = np.expand_dims(in_sequence, axis=0)
        # out_sequence = np.expand_dims(out_sequence, axis=0)

        return {'input':  torch.from_numpy(in_sequence),
                'output': torch.from_numpy(out_sequence)}


# ============================================================
# defining function which manipulate the classes above
# ============================================================
def import_data_loaders(batch_size, sequence_len=20):
    """
    This function imports the train and test database
    :param batch_size: size of each batch in the databases
    :param sequence_len: length of each sequence in the database
    :return: two datasets, training and validation
    """
    # --------------------------------------------------------
    # Importing complete dataset
    # --------------------------------------------------------
    word_database  = WordDataset()
    # train_database = WordSequenceDataset(path_train_data, word_database, sequence_len=sequence_len,
    #                                      transform=ToTensor())
    # valid_database = WordSequenceDataset(path_val_data, word_database, sequence_len=sequence_len,
    #                                      transform=ToTensor())
    train_database = WordSequenceDatasetIterable(path_train_data, word_database, sequence_len=sequence_len,
                                                 transform=ToTensor())
    valid_database = WordSequenceDatasetIterable(path_val_data, word_database, sequence_len=sequence_len,
                                                 transform=ToTensor())

    # --------------------------------------------------------
    # Creating the loaders
    # --------------------------------------------------------
    train_loader = DataLoader(train_database, batch_size=batch_size, drop_last=True, num_workers=4)
    test_loader  = DataLoader(valid_database, batch_size=batch_size, drop_last=True, num_workers=4)

    return train_loader, test_loader


# ============================================================
# help function, removing some words
# ============================================================
def remove_unnecessary_words(data, unnecessary=["<unk>", "N", "$", ""]):
    """
    :param data: a dataset listed as sentences
    :param unnecessary: a list of unwanted words
    :return: the function returns a list without the unwanted words
    """
    return [word for word in data if word not in unnecessary]


# **********************************************************************************************************************
# THIS SECTION DEALS WITH CLASSES REGARDING WITH THE NETWORKS
# **********************************************************************************************************************
# ============================================================
# LSTM network
# ============================================================
class NetLSTM(nn.Module):
    def __init__(self, word_database, device, hidden_num=200, layer_num=2, embed_size=1000, drop_prob=0.5):
        # --------------------------------------------------------
        # Allocating the properties of the class
        # --------------------------------------------------------
        super().__init__()
        self.hidden_num = hidden_num
        self.layer_num  = layer_num
        self.embed_num  = embed_size
        self.drop_prob  = drop_prob
        self.vocab      = word_database.vocab
        self.word2int   = word_database.word2int
        self.int2word   = word_database.int2word

        # --------------------------------------------------------
        # Allocating layers
        # --------------------------------------------------------
        self.embed   = nn.Embedding(len(word_database), embed_size)
        self.lstm    = nn.LSTM(embed_size, hidden_size=self.hidden_num, num_layers=self.layer_num,
                               dropout=self.drop_prob, batch_first=True)
        self.fc      = nn.Linear(self.hidden_num, len(word_database))

        # --------------------------------------------------------
        # Misc
        # --------------------------------------------------------
        self.device = device

    def forward(self, x, hidden):
        # --------------------------------------------------------
        # Embedding
        # --------------------------------------------------------
        output = self.embed(x)
        # --------------------------------------------------------
        # Passing through the LSTM
        # --------------------------------------------------------
        output, hidden = self.lstm(output, hidden)

        # --------------------------------------------------------
        # Flattening the tensor to 1D and passing through the FC
        # --------------------------------------------------------
        output = output.contiguous().view(-1, self.hidden_num)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size=20):
        # --------------------------------------------------------
        # creating an initial hidden tensor, set to all zeros
        # --------------------------------------------------------
        weight = next(self.parameters()).data
        hidden = (weight.new(self.layer_num, batch_size, self.hidden_num).zero_().to(self.device),
                  weight.new(self.layer_num, batch_size, self.hidden_num).zero_().to(self.device))

        return hidden


# ============================================================
# GRU network
# ============================================================
class NetGRU(nn.Module):
    def __init__(self, word_database, device, hidden_num=200, layer_num=2, embed_size=1000, drop_prob=0.5):
        # --------------------------------------------------------
        # Allocating the properties of the class
        # --------------------------------------------------------
        super().__init__()
        self.hidden_num = hidden_num
        self.layer_num = layer_num
        self.embed_num = embed_size
        self.drop_prob = drop_prob
        self.vocab     = word_database.vocab
        self.word2int  = word_database.word2int
        self.int2word  = word_database.int2word

        # --------------------------------------------------------
        # Allocating layers
        # --------------------------------------------------------
        self.embed = nn.Embedding(len(word_database), embed_size)
        self.gru = nn.GRU(embed_size, hidden_size=self.hidden_num, num_layers=self.layer_num,
                          dropout=self.drop_prob, batch_first=True)
        self.fc      = nn.Linear(self.hidden_num, len(word_database))

        # --------------------------------------------------------
        # Misc
        # --------------------------------------------------------
        self.device = device

    def forward(self, x, hidden):
        # --------------------------------------------------------
        # Embedding
        # --------------------------------------------------------
        output = self.embed(x)

        # --------------------------------------------------------
        # Passing through the GRU
        # --------------------------------------------------------
        output, hidden = self.gru(output, hidden)

        # --------------------------------------------------------
        # Flattening the tensor to 1D and passing through the FC
        # --------------------------------------------------------
        output = output.contiguous().view(-1, self.hidden_num)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size=20):
        # --------------------------------------------------------
        # creating an initial hidden tensor, set to all zeros
        # --------------------------------------------------------
        weight = next(self.parameters()).data
        hidden = weight.new(self.layer_num, batch_size, self.hidden_num).zero_().to(self.device)

        return hidden


class Trainer:
    def __init__(self, net, lr=0.01, grad_clip=5, sched_step=4, sched_gamma=0.5):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        # -------------------------------------
        # optimizer
        # -------------------------------------
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_step, gamma=sched_gamma)

        # -------------------------------------
        # Initializing the start epoch to zero
        # if not None, the model is pre-trained
        # -------------------------------------
        self.start_epoch = 0

        # -------------------------------------
        # Misc training parameters
        # -------------------------------------
        self.loss          = []
        self.learning_rate = lr
        self.grad_clip     = grad_clip

    def train(self, net, train_loader, test_loader, logger, batch_size=20, sequence_len=20, save_per_epochs=1):
        """
        :param net: Net class object, which is the net we want to train
        :param train_loader: holds the training database
        :param test_loader: holds the testing database
        :param logger: logging the results
        :param batch_size: the size of each batch in the training process
        :param sequence_len: the length of each sequence in the database
        :param save_per_epochs: flag indicating if you want to save
        :return: the function trains the network, and saves the trained network
        """
        losses             = []
        perplexities_train = []
        perplexities_val   = []
        print("Started training, learning rate: {}".format(self.learning_rate))
        # ----------------------------------------------------------
        # drop-out and batch normalization  behave differently in
        # training and evaluation, thus we use the following:
        # ----------------------------------------------------------
        net.train()
        for epoch in range(self.start_epoch, EPOCH_NUM):
            # ++++++++++++++++++++++++++++++++++++++++++
            # Initializing the hidden state
            # ++++++++++++++++++++++++++++++++++++++++++
            hidden = net.init_hidden(batch_size)
            train_loss = 0.0
            for ii, sample_batch in enumerate(train_loader):
                # ++++++++++++++++++++++++++++++++++++++
                # Extracting the in_seq and out_seq
                # ++++++++++++++++++++++++++++++++++++++
                in_seq  = sample_batch[0].long().to(net.device)
                out_seq = sample_batch[1].float().to(net.device)

                # in_seq = F.one_hot(in_seq, num_classes=len(train_loader.dataset.vocab))
                # out_seq = out_seq.float().to(net.device)

                # ++++++++++++++++++++++++++++++++++++++
                # Feed forward
                # ++++++++++++++++++++++++++++++++++++++
                self.optimizer.zero_grad()

                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # Generating new hidden vector to avoid
                # back-propagating through all the
                # sequence
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                if isinstance(net, NetLSTM):
                    hidden = tuple([tup.data for tup in hidden])
                elif isinstance(net, NetGRU):
                    hidden = hidden.data

                outputs, hidden = net(in_seq, hidden)
                # outputs = outputs.view(batch_size, sequence_len, len(net.vocab))
                out_seq = out_seq.view(-1).long()
                # ++++++++++++++++++++++++++++++++++++++
                # Computing the loss
                # ++++++++++++++++++++++++++++++++++++++
                loss = self.criterion(outputs, out_seq)

                # ++++++++++++++++++++++++++++++++++++++
                # Back propagation
                # ++++++++++++++++++++++++++++++++++++++
                loss.backward()
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # gradient clipping avoids divergence
                # thus we clip the gradient
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                self.optimizer.step()

                # ++++++++++++++++++++++++++++++++++++++
                # Documenting the loss
                # ++++++++++++++++++++++++++++++++++++++
                losses.append(loss.data.item())
                train_loss += loss.item() * in_seq.size(0)

                if ii == 60:
                    break

            self.loss = train_loss / len(train_loader)
            # ==========================================
            # Testing accuracy at the end of the epoch
            # ==========================================
            # perplexities_train.append(torch.pow(2, torch.tensor(self.loss)).item())
            perplexities_train.append(torch.exp(torch.tensor(self.loss)).item())
            perplexities_val.append(get_perplexity(net, self.criterion, test_loader, batch_size))

            # ==========================================
            # Advancing the scheduler of the lr
            # ==========================================
            self.scheduler.step()

            # ==========================================
            # Documenting with tensorboard
            # ==========================================
            logger.logger.add_scalars(logger.logger_tag + "_accuracy",
                                      {"Train_accuracy_learning_rate_{}".format(self.learning_rate): perplexities_train[-1],
                                       "Test_accuracy_learning_rate_{}".format(self.learning_rate): perplexities_val[-1]},
                                      epoch + 1)
            logger.logger.add_scalars(logger.logger_tag + "_loss",
                                      {"learning_rate_{}".format(self.learning_rate): self.loss},
                                      epoch + 1)

            # ==========================================
            # Saving the training state
            # save every x epochs and on the last epoch
            # ==========================================
            if epoch % save_per_epochs == 1 or epoch + 1 == EPOCH_NUM:
                save_state_train(self, filename=os.path.join("{}".format(logger.logger_tag),
                                                             "lr_{}_epoch_{}.tar".format(self.learning_rate, epoch+1)),
                                 net=net, epoch=epoch+1, lr=self.learning_rate, loss=self.loss)

            # ==========================================
            # Printing log to screen
            # ==========================================
            print("Epoch: {}/{} \tTraining loss: {:.6f} \tTrain perplexity: {:.6f} \tTest perplexity: {:.6f}".format(
                epoch + 1, EPOCH_NUM, self.loss,
                perplexities_train[-1], perplexities_val[-1]))


# **********************************************************************************************************************
# THIS SECTION DEALS WITH CLASSES REGARDING WITH THE logging
# **********************************************************************************************************************
class Logger:
    def __init__(self, logdir):
        # -------------------------------------
        # tensorboard logger
        # -------------------------------------
        self.logger = SummaryWriter(logdir)
        self.logger_tag = []


# **********************************************************************************************************************
# Class functions
# **********************************************************************************************************************
def get_perplexity(net, criterion, data_loader, batch_size=20):
    """
    :param net: the net upon which the perplexity is tested
    :param criterion: the criterion for the output - the cost function
    :param data_loader: the database used to test the perplexity
    :param batch_size: self explanatory
    :return: The function computes the perplexity, and returns it as a tensor value
    """
    # --------------------------------------------------------------------------------------------------------------
    # Internal variables
    # --------------------------------------------------------------------------------------------------------------
    one_hot_length = len(data_loader.dataset.vocab)
    losses         = []
    net.eval()

    # --------------------------------------------------------------------------------------------------------------
    # Beginning the iterations
    # --------------------------------------------------------------------------------------------------------------
    # ++++++++++++++++++++++++++++++++++++++++++
    # Initializing the hidden state
    # ++++++++++++++++++++++++++++++++++++++++++
    hidden = net.init_hidden(batch_size)
    with torch.no_grad():
        for ii, sample_batch in enumerate(data_loader):
            # ++++++++++++++++++++++++++++++++++++++
            # Extracting the in_seq and out_seq
            # ++++++++++++++++++++++++++++++++++++++
            in_seq = sample_batch[0].long().to(net.device)
            out_seq = sample_batch[1].float().to(net.device)

            # in_seq = F.one_hot(in_seq, num_classes=len(train_loader.dataset.vocab))
            # out_seq = out_seq.float().to(net.device)

            # ++++++++++++++++++++++++++++++++++++++
            # Feed forward
            # ++++++++++++++++++++++++++++++++++++++
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Generating new hidden vector to avoid
            # back-propagating through all the
            # sequence
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            if isinstance(net, NetLSTM):
                hidden = tuple([tup.data.to(net.device) for tup in hidden])
            elif isinstance(net, NetGRU):
                hidden = hidden.data.to(net.device)

            outputs, hidden = net(in_seq, hidden)
            # outputs = outputs.view(batch_size, sequence_len, len(net.vocab))
            out_seq = out_seq.view(-1).long()
            # ++++++++++++++++++++++++++++++++++++++
            # Computing the loss
            # ++++++++++++++++++++++++++++++++++++++
            loss = criterion(outputs, out_seq)
            losses.append(loss.item())

            if ii == 10:
                break

    # --------------------------------------------------------------------------------------------------------------
    # Computing the perplexity
    # --------------------------------------------------------------------------------------------------------------
    # perplexity = torch.pow(2, torch.tensor(np.mean(losses))).item()
    perplexity = torch.exp(torch.tensor(np.mean(losses))).item()

    net.train()
    return perplexity


def train_specific_mode(logger, device, train_loader, test_loader, word_database, mode=None, dropout=True):
    # ----------------------------------------------------------------------------
    # Test Parameters
    # ----------------------------------------------------------------------------
    # Possible modes: LSTM , GRU
    if mode is None:
        mode = LSTM

    print("Starting training for: {}, Dropout: {}".format(modes_dict[mode], dropout))
    # ----------------------------------------------------------------------------
    # Creating the net and allocating computation device
    # ----------------------------------------------------------------------------
    if mode == LSTM:
        net = NetLSTM(word_database, device, hidden_num=HIDDEN, layer_num=H_LAYER_NUM,
                      embed_size=EMBED_SIZE, drop_prob=DROPOUT_RATE)
    elif mode == GRU:
        net = NetGRU(word_database, device, hidden_num=HIDDEN, layer_num=H_LAYER_NUM,
                     embed_size=EMBED_SIZE, drop_prob=DROPOUT_RATE)
    net.to(device)  # allocating the computation to the CPU or GPU

    # ----------------------------------------------------------------------------
    # Creating the trainer object, and adding the logger tag
    # ----------------------------------------------------------------------------
    trainer = Trainer(net, MU, GRAD_CLIP, SCHEDULER_STEP, SCHEDULER_GAMMA)
    logger.logger_tag = modes_dict[mode]

    # ----------------------------------------------------------------------------
    # Initialize training
    # ----------------------------------------------------------------------------
    trainer.train(net, train_loader, test_loader, logger, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    # # ============================================================
    # # Debug step #1
    # # ============================================================
    # word_database  = WordDataset()
    # train_database = WordSequenceDataset(path_train_data, word_database, sequence_len=SEQUENCE_LENGTH,
    #                                      transform=ToTensor())
    #
    # for ii in range(len(train_database)):
    #     obj = train_database.__getitem__(ii)
    #     print('{}, {} , {}'.format(ii, len(obj['input']), len(obj['output'])))
    # # ============================================================
    # # Debug step #2
    # # ============================================================
    # train_loader, test_loader = import_data_sets(BATCH_SIZE)
    # for ii, sample_batch in enumerate(test_loader):
    #     in_sequences  = sample_batch['input']
    #     out_sequences = sample_batch['output']
    #     print('{}, {} , {}'.format(ii, len(in_sequences), len(out_sequences)))
    # # ============================================================
    # # Debug step #3
    # # ============================================================
    # word_database  = WordDataset()
    # train_database = WordSequenceDataset(path_train_data, word_database, sequence_len=SEQUENCE_LENGTH,
    #                                      transform=ToTensor())
    # print(dir(train_database))
    pass
