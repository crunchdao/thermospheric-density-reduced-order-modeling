import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy import fftpack
from pathlib import Path
# from pyJoules.energy_meter import measure_energy

import keras_tuner

mpl.use('Agg')

density_df_400_2013 = pd.read_csv('Data/2013_HASDM_500-575KM.den', delim_whitespace=True,
                                  header=None)
density_np_400_2013 = pd.DataFrame.to_numpy(density_df_400_2013)
del density_df_400_2013

# density_df_400_2014 = pd.read_csv('Data/2014_HASDM_400-475KM.den', delim_whitespace=True,
#                                   header=None)
# density_np_400_2014 = pd.DataFrame.to_numpy(density_df_400_2014)
# del density_df_400_2014
# density_df_500_2013 = pd.read_csv('Data/2013_HASDM_500-575KM.den', delim_whitespace=True,
#                                   header=None)
# density_np_500_2013 = pd.DataFrame.to_numpy(density_df_500_2013)
# del density_df_500_2013

nt = 19
nphi = 24

t = np.linspace(-np.pi / 2, np.pi / 2, nt)
phi = np.linspace(0, np.deg2rad(345), nphi)

# max_rho1 = np.max(density_np_400_2013[:, 10])
# max_rho2 = np.max(density_np_400_2014[:, 10])
# max_rho = np.max(np.array([max_rho1, max_rho2]))
# density_np_400_2013[:, 10] = density_np_400_2013[:, 10] / max_rho
# density_np_400_2014[:, 10] = density_np_400_2014[:, 10] / max_rho

max_rho = np.max(density_np_400_2013[:, 10])
density_np_400_2013[:, 10] = density_np_400_2013[:, 10] / max_rho

rho_list = []
rho_list1 = []
rho_list2 = []

for i in range(int(1331520 / (nt * nphi))):  # 1335168
    rho_400_i_2013 = density_np_400_2013[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
    rho_polar_400_i_2013 = np.reshape(rho_400_i_2013, (nt, nphi, 4))

    # rho_400_i_2014 = density_np_400_2013[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
    # rho_polar_400_i_2014 = np.reshape(rho_400_i_2014, (nt, nphi, 4))

    # rho_500_i = density_np_500_2013[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
    # rho_polar_500_i = np.reshape(rho_500_i, (nt, nphi, 4))
    #
    # rho_polar = np.concatenate(
    #     (rho_polar_400_i_2013, rho_polar_400_i_2014), axis=2)

    rho_polar_2013 = rho_polar_400_i_2013
    # rho_polar_2014 = rho_polar_400_i_2014

    rho_list1.append(rho_polar_2013)
    # rho_list2.append(rho_polar_2014)


# rho1 = np.array(rho_list1)
# rho2 = np.array(rho_list2)
# rho = np.concatenate((rho1, rho2), axis=0)  # Shape: (5840, 19, 24, 4)

rho = np.array(rho_list1)

rho_zeros = np.zeros((2920, 20, 24, 4))  # 5840, 20, 24, 4
rho_zeros[:, :nt, :nphi, :] = rho
del rho_list1, rho  #, rho_list2, rho1, rho2

training_data = rho_zeros[:2000]
validation_data = rho_zeros[2000:]

training_data_resh = np.reshape(training_data, newshape=(2000, 20*24*4))
nPoints_val = len(rho_zeros) - len(training_data)
validation_data_resh = np.reshape(validation_data, newshape=(nPoints_val, 20*24*4))

nPoints_val = 920  # 840
validation_data_resh = np.reshape(validation_data, newshape=(nPoints_val, 20*24*4))
rhoavg = np.mean(validation_data_resh, axis=0)  # Compute mean
rho_msub_val = validation_data_resh.T - np.tile(rhoavg, (nPoints_val, 1)).T  # Mean-subtracted data

# print(training_data_resh.shape)
rhoavg = np.mean(training_data_resh, axis=0)  # Compute mean
nPoints = 2000
rho_msub = training_data_resh.T - np.tile(rhoavg, (nPoints, 1)).T  # Mean-subtracted data
num_modes = 10


def build_model(hp):
    bottle = hp.Int("bottle", min_value=5, max_value=10)
    act = "relu"
    nlayers = hp.Int("num_layers", 1, 3)
    n_neurons = hp.Int("n_neurons", 4, 8)

    if nlayers == 1:
        model = keras.Sequential([
        layers.Flatten(),
        layers.Dense(20 * 24 * 4, activation=act),

        layers.Dense(2**n_neurons, activation=act),
        layers.Dense(units=bottle, activation=act),
        layers.Dense(2**n_neurons, activation=act),

        layers.Dense(20 * 24 * 4, activation=act)
        ])
    elif nlayers == 2:
        model = keras.Sequential([
        layers.Flatten(),
        layers.Dense(20 * 24 * 4, activation=act),

        layers.Dense(2**(n_neurons+1), activation=act),
        layers.Dense(2**n_neurons, activation=act),
        layers.Dense(units=bottle, activation=act),
        layers.Dense(2**n_neurons, activation=act),
        layers.Dense(2**(n_neurons+1), activation=act),

        layers.Dense(20 * 24 * 4, activation=act)
        ])
    else:
        model = keras.Sequential([
        layers.Flatten(),
        layers.Dense(20 * 24 * 4, activation=act),

        layers.Dense(2**(n_neurons+2), activation=act),
        layers.Dense(2**(n_neurons+1), activation=act),
        layers.Dense(2**n_neurons, activation=act),
        layers.Dense(units=bottle, activation=act),
        layers.Dense(2**n_neurons, activation=act),
        layers.Dense(2**(n_neurons+1), activation=act),
        layers.Dense(2**(n_neurons+2), activation=act),

        layers.Dense(20 * 24 * 4, activation=act)
        ])

    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    return model

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=90)

tuner.search(training_data_resh, training_data_resh, epochs=200, validation_data=(validation_data_resh, validation_data_resh))
best_model = tuner.get_best_models()[0]



# Vahid, these is the model for you:

# Value             |Best Value So Far |Hyperparameter
# 7                 |5                 |bottle
# 2                 |3                 |num_layers
# 8                 |8                 |n_neurons

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
                layers.Flatten(),
                layers.Dense(20 * 24 * 4, activation='relu'),

                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(5, activation='relu')           
        ])
        self.decoder = tf.keras.Sequential([
                layers.Input(shape=(5)),
                layers.Dense(256, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),

                layers.Dense(20 * 24 * 4, activation='relu')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



# @measure_energy
# def autoencoder_script():
    # class Autoencoder(Model):
    #     def __init__(self, z_size):
    #         super(Autoencoder, self).__init__()
    #         self.z_size = z_size
    #         self.encoder = tf.keras.Sequential([
    #                 layers.Flatten(),
    #                 layers.Dense(20 * 24 * 4, activation='relu'),
    #                 layers.Dense(512, activation='relu'),
    #                 layers.Dense(128, activation='relu'),
    #                 # layers.Dense(64, activation='tanh'),
    #                 layers.Dense(32, activation='relu'),
    #                 # layers.Dense(16, activation='relu'),
    #                 layers.Dense(z_size, activation='relu')           
    #         ])
    #         self.decoder = tf.keras.Sequential([
    #                 layers.Input(shape=(z_size)),
    #                 layers.Dense(32, activation='relu'),
    #                 layers.Dense(128, activation='relu'),
    #                 layers.Dense(512, activation='relu'),
    #                 layers.Dense(20 * 24 * 4, activation='relu')
    #         ])

    #     def call(self, x):
    #         encoded = self.encoder(x)
    #         decoded = self.decoder(encoded)
    #         return decoded


#     # print(X_pca.shape)      
#     # print(X_pca_val.shape)
#     # print(training_data_resh.shape)
#     # print(validation_data_resh.shape)
#     # exit()
#     # csv_handler = CSVHandler('result.csv')

#     # @measure_energy()
#     # def foo():

#     num_modes = 10
#     run = True
#     if run:
#         autoencoder = Autoencoder(num_modes)
#         autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#         history = autoencoder.fit(training_data_resh, training_data_resh,
#                         batch_size=5,  # play with me
#                         epochs= 200,  # 200
#                         shuffle=True,
#                         validation_data=(validation_data_resh, validation_data_resh))

#         loss = list(history.history.values())
#         autoencoder.encoder.save('encoder')
#         autoencoder.decoder.save('decoder')
#         encoded = autoencoder.encoder(validation_data_resh).numpy()
#         decoded = autoencoder.decoder(encoded).numpy()
#     else:
#         encoder = tf.keras.models.load_model('encoder')
#         decoder = tf.keras.models.load_model('decoder')
#         encoded = encoder(validation_data_resh).numpy()
#         decoded = decoder(encoded).numpy()

# autoencoder_script()


# np.savetxt('output/atm/encoded.txt', encoded, delimiter=',')
# np.savetxt('output/atm/training_data.txt', training_data_resh, delimiter=',')
# np.savetxt('output/atm/validation_data.txt', validation_data_resh, delimiter=',')
# np.savetxt('output/atm/decoded.txt', decoded, delimiter=',')
# if run:
#     plt.figure()
#     plt.rcParams.update({'font.size': 14})  # increase the font size
#     mpl.rcParams['legend.fontsize'] = 15
#     plt.xlabel("Number of Epoch")
#     plt.ylabel("Loss")
#     plt.plot(loss[0], label="Train", linewidth=2)
#     plt.plot(loss[1], label="Validation", linewidth=2)
#     plt.yscale("log")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('output/loss_atm.png')
# error = decoded - validation_data_resh 
# error = np.reshape(error, newshape=(nPoints_val, 20, 24, 4))
# plt.figure() 
# plt.rcParams.update({'font.size': 14})  # increase the font size
# mpl.rcParams['legend.fontsize'] = 15
# plt.xlabel("Longitude [deg]")
# plt.ylabel("Latitude [deg]")
# plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[10, :19, :, 0])/validation_data[10, :19, :, 0]*100,
#                 cmap="inferno", levels=900)
# plt.colorbar()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('output/ReconstructionError_nn.png')
# error_norm_nn = linalg.norm(decoded - validation_data_resh)  # , ord=inf
# print('error_norm_nn:', error_norm_nn)

# # foo()
# # csv_handler.save_data()

