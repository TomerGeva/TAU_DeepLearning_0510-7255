import torch
from functions import import_data_sets, accuracy_test, save_state_train
from classes import train_specific_mode, load_state_train, generate_fake_images
from classes import Logger, Trainer
from Config import *


def main():
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = Logger(PATH_LOGS)

    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ================================================================================
    # Fetching the train and test data
    # ================================================================================
    train_loader, test_loader = import_data_sets(BATCH_SIZE)

    # ================================================================================
    # Finding optimal learning rate
    # ================================================================================
    if FIND_OPTIMAL_LEARNING_RATE:
        # Possible modes: NO_REGULARIZATION , DROP_OUT , WEIGHT_DECAY , BATCH_NORMALIZATION
        mode = DCGAN
        find_optimal_learning_rate(logger, device, train_loader, test_loader, mode)

    # ================================================================================
    # Iterating through the modes of the networks
    # ================================================================================
    if PER_MODE_CONVERGENCE:
        per_mode_convergence(logger, device, train_loader, test_loader)

    # ================================================================================
    # Training one mode alone
    # ================================================================================
    if TRAIN_SPECIFIC_MODE:
        # Possible modes: DCGAN, ...
        mode = DCGAN
        train_specific_mode(logger, device, train_loader, mode)

    # ================================================================================
    # Loading a trained model and generating pictures
    # ================================================================================
    if LOAD_AND_TEST:
        # Possible modes: DCGAN, WGAN
        mode = DCGAN
        # ----------------------------------------------------------------------------
        # Loading the network and trainer
        # ----------------------------------------------------------------------------
        gen, disc, trainer = load_state_train(mode)
        gen.eval()
        disc.eval()

        # ================================================================================
        # Generating fake images
        # ================================================================================
        generate_fake_images(gen)

    logger.logger.close()


if __name__ == '__main__':
    main()
