from Config import *
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


def import_data_sets(batch_size):
    """
    :param batch_size: the size of each batch to run through the network
    :return: this function returns the train database and test database, after organizing it as loaders
    """
    # -------------------------------------------------------------------
    # Transformations - defining useful transformations
    # -------------------------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize((X_SIZE, Y_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([TRANSFORM_NORM] * IMG_CHANNELS, [TRANSFORM_NORM] * IMG_CHANNELS)
        ])

    # -------------------------------------------------------------------
    # downloading the relevant datasets
    # -------------------------------------------------------------------
    train_data = datasets.FashionMNIST(root='data', download=True, train=True, transform=transform)
    test_data = datasets.FashionMNIST(root='data', download=True, train=False, transform=transform)

    # -------------------------------------------------------------------
    # preparing the data loaders
    # -------------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def show_batch(train_loader, batch_size, labels_dict):
    """
    :param train_loader: the loader of the train database, showing examples
    :param batch_size: the size of a batch
    :param labels_dict: the label dictionary
    :return: this function plots the batch, for us to see the database
    """
    data_iteration = iter(train_loader)
    images, labels = data_iteration.next()
    images = images.numpy()

    fig = plt.figure(figsize=(10, 10))
    row_num = 8
    for ii in range(batch_size):
        # Start next subplot.
        plt.subplot(row_num, batch_size / row_num, ii + 1, title=labels_dict[(labels[ii].item())])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(images[ii]), cmap=plt.cm.binary)

    plt.show()


def accuracy_test(epoch, net, loader):
    # BATCH_SIZE = BATCH_SIZE
    correct = 0.0
    total = 0.0
    i = epoch * BATCH_SIZE
    for images, labels in loader:
        images = Variable(images.float()).to(net.device)
        labels = labels.to(net.device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if i == (epoch+1) * BATCH_SIZE:
            break
    return (100 * correct / total).item()


def save_state_train(trainer, filename, gen, disc, epoch, lr):
    """Saving model and optimizer to drive, as well as current epoch and loss
    # When saving a general checkpoint, to be used for either inference or resuming training, you must save more
    # than just the model’s state_dict.
    # It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are
    # updated as the model trains.
    """
    path = os.path.join(PATH_MODELS, filename)
    data_to_save = {'epoch': epoch,
                    'gen_state_dict': gen.state_dict(),
                    'disc_state_dict': disc.state_dict(),
                    'generator_optimizer_state_dict': trainer.optimizer_gen.state_dict(),
                    'discriminator_optimizer_state_dict': trainer.optimizer_disc.state_dict(),
                    'lr': lr
                    }
    torch.save(data_to_save, path)


# if __name__ == "__main__":
#     train_loader, test_loader = import_data_sets(BATCH_SIZE)
#     show_batch(train_loader, BATCH_SIZE, labels_dict)
