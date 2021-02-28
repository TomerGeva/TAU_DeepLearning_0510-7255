# from functions import *
from Classes import *
from Config import *


def main():
    # ================================================================================
    # Setting the logger and the word database
    # ================================================================================
    logger = Logger(path_logs)
    word_database = WordDataset()

    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ================================================================================
    # Fetching the train and test data
    # ================================================================================
    train_loader, test_loader = import_data_loaders(BATCH_SIZE, SEQUENCE_LENGTH)

    # ================================================================================
    # Training one mode alone
    # ================================================================================
    if TRAIN_SPECIFIC_MODE:
        # Possible modes: LSTM, GRU
        mode = LSTM
        train_specific_mode(logger, device, train_loader, test_loader, word_database, mode=mode, dropout=DROPOUT)

    logger.logger.close()


if __name__ == '__main__':
    main()
