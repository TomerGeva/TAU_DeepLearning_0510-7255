from functions import *
from classes import *
from Config import *


def main():
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = Logger(path_logs)

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
        mode = BATCH_NORMALIZATION
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
        # Possible modes: NO_REGULARIZATION , DROP_OUT , WEIGHT_DECAY , BATCH_NORMALIZATION
        mode = BATCH_NORMALIZATION
        train_specific_mode(logger, device, train_loader, test_loader, mode)

    # ================================================================================
    # Loading a trained model and testing the accuracy
    # ================================================================================
    if LOAD_AND_TEST:
        # Possible modes: NO_REGULARIZATION , DROP_OUT , WEIGHT_DECAY , BATCH_NORMALIZATION
        mode = BATCH_NORMALIZATION
        # ----------------------------------------------------------------------------
        # Loading the network and trainer
        # ----------------------------------------------------------------------------
        net, trainer = load_state_train(mode)
        net.eval()

        # ================================================================================
        # Testing accuracy
        # ================================================================================
        train_accuracy = accuracy_test(0, net, train_loader)
        test_accuracy = accuracy_test(0, net, test_loader)
        # ================================================================================
        # Printing the results
        # ================================================================================
        print("Mode: {} \tTrain accuracy: {:.6f}% \tTest accuracy: {:.6f}%".format(
            modes_dict[mode], train_accuracy, test_accuracy))

    logger.logger.close()


if __name__ == '__main__':
    main()
