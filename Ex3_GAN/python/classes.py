from Config import *
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from functions import accuracy_test, save_state_train


class DiscriminatorDCGAN(nn.Module):
    def __init__(self, channels_img, feature_list, kernel_list, stride_list, padding_list, alpha):
        """
        :param channels_img: number of channels in the input image, should be 1
        :param feature_list: list of features in each layer of the conv2D
        :param kernel_list: kernel size used in each Conv2d layer
        :param stride_list: stride in each Conv2d layer
        :param padding_list: padding in each Conv2d layer
        :param alpha: leaky ReLU parameter
        """
        super(DiscriminatorDCGAN, self).__init__()
        self.disc = nn.Sequential(
            # --------------------------------------------------------------------------
            # First layer should be without batch norm, thus inserted manually
            # --------------------------------------------------------------------------
            nn.Conv2d(channels_img,
                      feature_list[0],
                      kernel_size=kernel_list[0],
                      stride=stride_list[0],
                      padding=padding_list[0],
                      ),
            nn.LeakyReLU(alpha),
            # --------------------------------------------------------------------------
            # Second layer
            # --------------------------------------------------------------------------
            self._block(feature_list[0],
                        feature_list[1],
                        kernel_size=kernel_list[1],
                        stride=stride_list[1],
                        padding=padding_list[1],
                        alpha=alpha,
                        ),
            # --------------------------------------------------------------------------
            # Third layer
            # --------------------------------------------------------------------------
            self._block(feature_list[1],
                        feature_list[2],
                        kernel_size=kernel_list[2],
                        stride=stride_list[2],
                        padding=padding_list[2],
                        alpha=alpha,
                        ),
            # --------------------------------------------------------------------------
            # Fourth layer
            # --------------------------------------------------------------------------
            self._block(feature_list[2],
                        feature_list[3],
                        kernel_size=kernel_list[3],
                        stride=stride_list[3],
                        padding=padding_list[3],
                        alpha=alpha,
                        ),
            # --------------------------------------------------------------------------
            # Fifth layer, single output
            # --------------------------------------------------------------------------
            nn.Conv2d(feature_list[3],
                      1,
                      kernel_size=kernel_list[4],
                      stride=stride_list[4],
                      padding=padding_list[4],
                      ),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, alpha):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False,
                      ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        return self.disc(x)


class GeneratorDCGAN(nn.Module):
    def __init__(self, z_dim, channels_img, feature_list, kernel_list, stride_list, padding_list):
        """
        :param z_dim: noise dimension vector
        :param channels_img: number of channels in the input image, should be 1
        :param feature_list: list of features in each layer of the conv2D
        :param kernel_list: kernel size used in each Conv2d layer
        :param stride_list: stride in each Conv2d layer
        :param padding_list: padding in each Conv2d layer
        """
        super(GeneratorDCGAN, self).__init__()
        self.gen = nn.Sequential(
            # --------------------------------------------------------------------------
            # First layer
            # --------------------------------------------------------------------------
            self._block(z_dim,
                        feature_list[0],
                        kernel_size=kernel_list[0],
                        stride=stride_list[0],
                        padding=padding_list[0],
                        ),
            # --------------------------------------------------------------------------
            # Second layer
            # --------------------------------------------------------------------------
            self._block(feature_list[0],
                        feature_list[1],
                        kernel_size=kernel_list[1],
                        stride=stride_list[1],
                        padding=padding_list[1],
                        ),
            # --------------------------------------------------------------------------
            # Third layer
            # --------------------------------------------------------------------------
            self._block(feature_list[1],
                        feature_list[2],
                        kernel_size=kernel_list[2],
                        stride=stride_list[2],
                        padding=padding_list[2],
                        ),
            # --------------------------------------------------------------------------
            # Fourth layer
            # --------------------------------------------------------------------------
            self._block(feature_list[2],
                        feature_list[3],
                        kernel_size=kernel_list[3],
                        stride=stride_list[3],
                        padding=padding_list[3],
                        ),
            # --------------------------------------------------------------------------
            # Fifth layer, without batch norm
            # --------------------------------------------------------------------------
            nn.ConvTranspose2d(feature_list[3],
                               channels_img,
                               kernel_size=kernel_list[4],
                               stride=stride_list[4],
                               padding=padding_list[4],
                               ),
            nn.Tanh(),  # making the output to be between -1 and 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding,
                               bias=False,
                               ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


class Logger:
    def __init__(self, logdir):
        # -------------------------------------
        # tensorboard logger
        # -------------------------------------
        self.logger = SummaryWriter(logdir)
        self.images_real = SummaryWriter(os.path.join(logdir, 'images'))
        self.images_fake = SummaryWriter(os.path.join(logdir, 'images'))
        self.logger_tag  = []


class Trainer:
    def __init__(self, gen, disc, lr=2e-4, betas=(0.5, 0.999), grad_clip=5):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.criterion = nn.BCEWithLogitsLoss()
        # -------------------------------------
        # optimizers
        # -------------------------------------
        self.optimizer_gen  = optim.Adam(gen.parameters(), lr=lr, betas=betas)
        self.optimizer_disc = optim.Adam(disc.parameters(), lr=lr, betas=betas)

        # -------------------------------------
        # Initializing the start epoch to zero
        # if not None, the model is pre-trained
        # -------------------------------------
        self.start_epoch = 0

        # -------------------------------------
        # Misc training parameters
        # -------------------------------------
        self.learning_rate = lr
        self.betas         = betas
        self.grad_clip     = grad_clip

    def train(self, gen, disc, train_loader, logger, mode, device=None, save_per_epochs=1):
        """
        :param gen: generator we want to train
        :param disc: discriminator we want to train
        :param train_loader: holds the training database
        :param logger: logging the results
        :param device: allocation of computation
        :param save_per_epochs: flag indicating if you want to save
        :return: the function trains the network, and saves the trained network
        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
        print("Started training, learning rate: {}".format(self.learning_rate))
        iteration = 0  # discriminator input iteration
        step      = 0  # fixed_noise example pictures generation
        # ------------------------------------------------------------------------------------------------
        # batch normalization behaves differently in training and evaluation, thus we use the following:
        # ------------------------------------------------------------------------------------------------
        gen.train()
        disc.train()
        for epoch in range(self.start_epoch, EPOCH_NUM):
            for ii, (real, _) in enumerate(train_loader):
                iteration += 1
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Allocating images and noise
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                real  = real.to(device)
                noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
                fake  = gen(noise)

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Training Discriminator & computing the loss
                #    max{ log(D(x)) + log(1-D(G(z)) }
                #          fake             real
                # In the real loss part we put ones in the target, since these are real pictures
                # In the fake loss part we put zeros in the target, since these are fake pictures
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                disc_real = disc(real).reshape(-1)
                disc_fake = disc(fake).reshape(-1)

                if mode == DCGAN:
                    loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
                    loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                elif mode == WGAN:
                    loss_disc_real = -1 * torch.mean(disc_real)
                    loss_disc_fake = torch.mean(disc_fake)

                loss_disc = (loss_disc_real + loss_disc_fake) / 2

                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # Back propagation
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                self.optimizer_disc.zero_grad()
                disc.zero_grad()
                loss_disc.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), self.grad_clip)
                self.optimizer_disc.step()

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Training Generator & computing the loss
                #    min{ log(1-D(G(z)) }  <----->  max{ log(G(z)) }
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                fake     = gen(noise)
                output   = disc(fake).reshape(-1)
                if mode == DCGAN:
                    loss_gen = self.criterion(output, torch.ones_like(output))
                elif mode == WGAN:
                    loss_gen = -1 * torch.mean(output)

                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # Back propagation
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                self.optimizer_gen.zero_grad()
                gen.zero_grad()
                loss_gen.backward()
                nn.utils.clip_grad_norm_(gen.parameters(), self.grad_clip)
                self.optimizer_gen.step()

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Documenting
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # Loss per iteration
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                logger.logger.add_scalars(logger.logger_tag + "_losses",
                                          {"Discriminator_learning_rate_{}".format(self.learning_rate): loss_disc,
                                           "Generator_learning_rate_{}".format(self.learning_rate): loss_gen,
                                           },
                                          iteration)

                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # Generating pictures for documentation
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                if ii % 100 == 0:
                    step += 1
                    with torch.no_grad():
                        fake = gen(fixed_noise)
                        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                        logger.images_real.add_image(logger.logger_tag + "real", img_grid_real, global_step=step)
                        logger.images_fake.add_image(logger.logger_tag + "fake", img_grid_fake, global_step=step)

            # ========================================================================================
            # Saving the training state
            # save every x epochs and on the last epoch
            # ========================================================================================
            if epoch % save_per_epochs == 1 or epoch + 1 == EPOCH_NUM:
                save_state_train(self,
                                 filename=os.path.join("{}".format(logger.logger_tag),
                                                       "lr_{}_epoch_{}.tar".format(self.learning_rate, epoch+1)),
                                 gen=gen,
                                 disc=disc,
                                 epoch=epoch+1,
                                 lr=self.learning_rate,
                                 )

            # ========================================================================================
            # Printing log to screen
            # ========================================================================================
            print("Epoch: {}/{} . . .".format(epoch + 1, EPOCH_NUM))


def initialize_weights(net, mean, std):
    """
    :param net: the model which is being normalized
    :param mean: the target mean of the weights
    :param std: the target standard deviation of the weights
    :return: nothing, just adjusts the weights
    """
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, mean, std)


def load_state_train(mode):
    """Loads training state from drive or memory
    when loading the dictionary, the function also arranges the data in such a manner which allows to
    continure the training
    """
    # -------------------------------------
    # assembling the path
    # -------------------------------------
    filename = r"lr_0.0002_epoch_25.tar"
    path = os.path.join(PATH_MODELS, MODES_DICT[mode], filename)

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
    gen = GeneratorDCGAN(Z_DIM, IMG_CHANNELS,
                         GEN_FEATURE_NUMS,
                         KERNEL_LIST,
                         GEN_STRIDE_LIST,
                         GEN_PADDING_LIST,
                         ).to(device)
    gen.load_state_dict(checkpoint['gen_state_dict'])

    disc = DiscriminatorDCGAN(IMG_CHANNELS,
                              DISC_FEATURE_NUMS,
                              KERNEL_LIST,
                              DISC_STRIDE_LIST,
                              DISC_PADDING_LIST,
                              ALPHA,
                              ).to(device)
    disc.load_state_dict(checkpoint['disc_state_dict'])

    trainer = Trainer(gen, disc, MU)
    trainer.learning_rate = checkpoint['lr']
    trainer.start_epoch   = checkpoint['epoch']

    trainer.optimizer_gen.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    trainer.optimizer_disc.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

    return gen, disc, trainer


def train_specific_mode(logger, device, train_loader, mode=None):
    # ----------------------------------------------------------------------------
    # Test Parameters
    # ----------------------------------------------------------------------------
    # Possible modes: DCGAN, WGAN
    if mode is None:
        mode = DCGAN

    print("Starting training for: {}".format(MODES_DICT[mode]))
    # ----------------------------------------------------------------------------
    # Creating the generator & discriminator + allocating computation device
    # ----------------------------------------------------------------------------
    gen  = GeneratorDCGAN(Z_DIM, IMG_CHANNELS,
                          GEN_FEATURE_NUMS,
                          KERNEL_LIST,
                          GEN_STRIDE_LIST,
                          GEN_PADDING_LIST,
                          ).to(device)
    disc = DiscriminatorDCGAN(IMG_CHANNELS,
                              DISC_FEATURE_NUMS,
                              KERNEL_LIST,
                              DISC_STRIDE_LIST,
                              DISC_PADDING_LIST,
                              ALPHA,
                              ).to(device)
    initialize_weights(gen, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD)
    initialize_weights(disc, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD)

    # ----------------------------------------------------------------------------
    # Creating the trainer object, and adding the logger tag
    # ----------------------------------------------------------------------------
    trainer = Trainer(gen, disc, MU)
    logger.logger_tag = MODES_DICT[mode]

    # ----------------------------------------------------------------------------
    # Initialize training
    # ----------------------------------------------------------------------------
    trainer.train(gen, disc, train_loader, logger, mode)
    

def generate_fake_images(gen, block_size=32, device=None):
    """
    :param gen: Generator
    :param block_size: number of fake images
    :param device: cpu or cuda
    :return: plots the fake images on screen
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # =================================================================================================
    # Generating noise
    # =================================================================================================
    noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
    with torch.no_grad():
        # =============================================================================================
        # Passing through the generator
        # =============================================================================================
        fake = gen(noise)
        # =============================================================================================
        # Plotting
        # =============================================================================================
        fake = fake.numpy()

        fig = plt.figure(figsize=(10, 10))
        plt.suptitle("Fake images, generated by the generator of the chosen model")
        row_num = 8
        for ii in range(block_size):
            # Start next subplot.
            plt.subplot(row_num, int(block_size / row_num), ii + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(np.squeeze(fake[ii]), cmap=plt.cm.binary)

        plt.show()

