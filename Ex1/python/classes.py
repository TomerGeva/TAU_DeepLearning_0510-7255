from Config import *
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from functions import *


class Net(nn.Module):
    def __init__(self, mode, device):
        super(Net, self).__init__()
        self.mode        = mode
        self.device      = device
        self.description = layer_description
        # ---------------------------------------------------------
        # constructing the first layer, as a function of the mode
        # ---------------------------------------------------------
        if mode == BATCH_NORMALIZATION:
            self.layer1 = nn.Sequential(nn.Conv2d(1, filter_num[0], kernel_size=kernel_sizes[0], padding=padding[0]),
                                        nn.BatchNorm2d(filter_num[0]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(max_pool_size))
        # ---------------------------------------------------------
        # if mode is dropout, using the dropout
        # ---------------------------------------------------------
        elif mode == DROP_OUT:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, filter_num[0], kernel_size=kernel_sizes[0], padding=padding[0]),
                nn.BatchNorm2d(filter_num[0]),
                nn.ReLU(),
                nn.MaxPool2d(max_pool_size),
                nn.Dropout(p=DROPOUT_RATE))
        else:
            self.layer1 = nn.Sequential(nn.Conv2d(1, filter_num[0], kernel_size=kernel_sizes[0], padding=padding[0]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(max_pool_size))

        # ---------------------------------------------------------
        # constructing the second layer, as a function of the mode
        # ---------------------------------------------------------
        if mode == BATCH_NORMALIZATION:
            self.layer2 = nn.Sequential(nn.Conv2d(filter_num[0], filter_num[1],
                                                  kernel_size=kernel_sizes[1], padding=padding[1]),
                                        nn.BatchNorm2d(filter_num[1]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(max_pool_size))
        else:
            self.layer2 = nn.Sequential(nn.Conv2d(filter_num[0], filter_num[1],
                                                  kernel_size=kernel_sizes[1], padding=padding[1]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(max_pool_size))
        # ---------------------------------------------------------
        # computing the dimensions of the convolution output
        # ---------------------------------------------------------
        x_dim, y_dim = self.compute_dim_sizes()  # should be 7
        self.fc1 = nn.Linear(x_dim * y_dim * filter_num[-1], fc_layers[0])
        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])

    def compute_dim_sizes(self):
        x_dim_size = X_SIZE
        y_dim_size = Y_SIZE
        counter    = 0
        for action in range(len(self.description)):
            if self.description[action] == 'conv':
                x_dim_size = int((x_dim_size - (kernel_sizes[counter] - strides[counter]) + 2*padding[counter])
                                 / strides[counter])
                y_dim_size = int((y_dim_size - (kernel_sizes[counter] - strides[counter]) + 2*padding[counter])
                                 / strides[counter])
                counter += 1
            elif self.description[action] == 'pool':
                x_dim_size = int(x_dim_size / max_pool_size)
                y_dim_size = int(y_dim_size / max_pool_size)

        return x_dim_size, y_dim_size

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Logger:
    def __init__(self, logdir):
        # -------------------------------------
        # tensorboard logger
        # -------------------------------------
        self.logger = SummaryWriter(logdir)
        self.logger_tag = []


class Trainer:
    def __init__(self, net, lr=0.01):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        # -------------------------------------
        # optimizer
        # -------------------------------------
        if net.mode == WEIGHT_DECAY:
            self.optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.01)
        else:
            self.optimizer = optim.SGD(net.parameters(), lr=lr)
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

    def train(self, net, train_loader, test_loader, logger, save_per_epochs=1):
        """
        :param net: Net class object, which is the net we want to train
        :param train_loader: holds the training database
        :param test_loader: holds the testing database
        :param logger: logging the results
        :param save_per_epochs: flag indicating if you want to save
        :return: the function trains the network, and saves the trained network
        """
        losses           = []
        accuracies_train = []
        accuracies_test  = []
        print("Started training, learning rate: {}".format(self.learning_rate))
        # ----------------------------------------------------------
        # drop-out and batch normalization  behave differently in
        # training and evaluation, thus we use the following:
        # ----------------------------------------------------------
        net.eval()
        for epoch in range(self.start_epoch, EPOCH_NUM):
            train_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                # ++++++++++++++++++++++++++++++++++++++
                # Extracting the images and labels
                # ++++++++++++++++++++++++++++++++++++++
                images = Variable(images.float()).to(net.device)
                labels = Variable(labels).to(net.device)

                # ++++++++++++++++++++++++++++++++++++++
                # Feed forward
                # ++++++++++++++++++++++++++++++++++++++
                self.optimizer.zero_grad()
                outputs = net(images)

                # ++++++++++++++++++++++++++++++++++++++
                # Computing the loss
                # ++++++++++++++++++++++++++++++++++++++
                loss = self.criterion(outputs, labels)

                # ++++++++++++++++++++++++++++++++++++++
                # Back propagation
                # ++++++++++++++++++++++++++++++++++++++
                loss.backward()
                self.optimizer.step()

                # ++++++++++++++++++++++++++++++++++++++
                # Documenting the loss
                # ++++++++++++++++++++++++++++++++++++++
                losses.append(loss.data.item())
                train_loss += loss.item() * images.size(0)

            self.loss = train_loss / len(train_loader.dataset)
            # ==========================================
            # Testing accuracy at the end of the epoch
            # ==========================================
            accuracies_train.append(accuracy_test(epoch, net, train_loader))
            accuracies_test.append(accuracy_test(epoch, net, test_loader))

            # ==========================================
            # Documenting with tensorboard
            # ==========================================
            logger.logger.add_scalars(logger.logger_tag + "_accuracy",
                                    {"Train_accuracy_learning_rate_{}".format(self.learning_rate): accuracies_train[-1],
                                     "Test_accuracy_learning_rate_{}".format(self.learning_rate): accuracies_test[-1]},
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
            print("Epoch: {}/{} \tTraining loss: {:.6f} \tTrain accuracy: {:.6f}% \tTest accuracy: {:.6f}%".format(
                epoch + 1, EPOCH_NUM, self.loss,
                accuracies_train[-1], accuracies_test[-1]))


def load_state_train(mode):
    """Loads training state from drive or memory
    when loading the dictionary, the function also arranges the data in such a manner which allows to
    continure the training
    """
    # -------------------------------------
    # assembling the path
    # -------------------------------------
    filename = r"lr_0.01_epoch_50.tar"
    path = os.path.join(path_models, modes_dict[mode], filename)

    # -------------------------------------
    # allocating device type
    # -------------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------
    # loading the dictionary
    # -------------------------------------
    checkpoint = torch.load(path, map_location=device)

    # -------------------------------------
    # arranging the data
    # -------------------------------------
    net     = Net(mode, device)
    net.to(device)  # allocating the computation to the CPU or GPU
    trainer = Trainer(net)

    trainer.loss          = checkpoint['loss']
    trainer.learning_rate = checkpoint['lr']
    trainer.start_epoch   = checkpoint['epoch']

    net.load_state_dict(checkpoint['model_state_dict'])
    if mode == WEIGHT_DECAY:
        trainer.optimizer = optim.SGD(net.parameters(), lr=trainer.learning_rate, weight_decay=0.01)
    else:
        trainer.optimizer = optim.SGD(net.parameters(), lr=trainer.learning_rate)
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return net, trainer


def find_optimal_learning_rate(logger, device, train_loader, test_loader, mode=None):
    # ----------------------------------------------------------------------------
    # Test Parameters
    # ----------------------------------------------------------------------------
    # Possible modes: NO_REGULARIZATION , DROP_OUT , WEIGHT_DECAY , BATCH_NORMALIZATION
    if mode is None:
        mode = BATCH_NORMALIZATION
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]

    # ----------------------------------------------------------------------------
    # Setting the logger
    # ----------------------------------------------------------------------------
    logger.logger_tag = modes_dict[mode]

    # ----------------------------------------------------------------------------
    # Running the test for the best learning rate
    # ----------------------------------------------------------------------------
    for lr in learning_rates:
        # ----------------------------------------------------------------------------
        # Creating the net and allocating computation device
        # ----------------------------------------------------------------------------
        net = Net(mode, device)
        net.to(device)  # allocating the computation to the CPU or GPU

        # ----------------------------------------------------------------------------
        # Creating the trainer object
        # ----------------------------------------------------------------------------
        trainer = Trainer(net, lr)

        # ----------------------------------------------------------------------------
        # Initialize training
        # ----------------------------------------------------------------------------
        trainer.train(net, train_loader, test_loader, logger)


def per_mode_convergence(logger, device, train_loader, test_loader):
    for mode in modes_dict:
        print("Starting training for: {}".format(modes_dict[mode]))
        # ----------------------------------------------------------------------------
        # Creating the net and allocating computation device
        # ----------------------------------------------------------------------------
        net = Net(mode, device)
        net.to(device)  # allocating the computation to the CPU or GPU

        # ----------------------------------------------------------------------------
        # Creating the trainer object, and adding the logger tag
        # ----------------------------------------------------------------------------
        trainer = Trainer(net, MU)
        logger.logger_tag = modes_dict[mode]

        # ----------------------------------------------------------------------------
        # Initialize training
        # ----------------------------------------------------------------------------
        trainer.train(net, train_loader, test_loader, logger)


def train_specific_mode(logger, device, train_loader, test_loader, mode=None):
    # ----------------------------------------------------------------------------
    # Test Parameters
    # ----------------------------------------------------------------------------
    # Possible modes: NO_REGULARIZATION , DROP_OUT , WEIGHT_DECAY , BATCH_NORMALIZATION
    if mode is None:
        mode = BATCH_NORMALIZATION

    print("Starting training for: {}".format(modes_dict[mode]))
    # ----------------------------------------------------------------------------
    # Creating the net and allocating computation device
    # ----------------------------------------------------------------------------
    net = Net(mode, device)
    net.to(device)  # allocating the computation to the CPU or GPU

    # ----------------------------------------------------------------------------
    # Creating the trainer object, and adding the logger tag
    # ----------------------------------------------------------------------------
    trainer = Trainer(net, MU)
    logger.logger_tag = modes_dict[mode]

    # ----------------------------------------------------------------------------
    # Initialize training
    # ----------------------------------------------------------------------------
    trainer.train(net, train_loader, test_loader, logger)
