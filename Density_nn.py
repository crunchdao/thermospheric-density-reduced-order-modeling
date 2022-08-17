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

import os

mpl.use('Agg')

years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
for year in years:
    print(year)
    density_df = pd.read_csv('./Data/{}_HASDM_500-575KM.txt'.format(year), delim_whitespace=True,
                                  header=None)
    
    density_np = pd.DataFrame.to_numpy(density_df)
    del density_df

    nt = 19
    nphi = 24

    t = np.linspace(-np.pi / 2, np.pi / 2, nt)
    phi = np.linspace(0, np.deg2rad(345), nphi)

    max_rho = np.max(density_np[:, 7])
    density_np[:, 7] = density_np[:, 7] / max_rho

    rho_list = []
    rho_list1 = []
    rho_list2 = []

    for i in range(int(1331520 / (nt * nphi))):  # 1335168
        rho_i = density_np[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 7]
        rho_polar_i = np.reshape(rho_i, (nt, nphi, 4))

        rho_list1.append(rho_polar_i)

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


# def build_model(hp):
#     bottle = hp.Int("bottle", min_value=5, max_value=10)
#     act = "relu"
#     nlayers = hp.Int("num_layers", 1, 3)
#     n_neurons = hp.Int("n_neurons", 4, 8)

#     if nlayers == 1:
#         model = keras.Sequential([
#         layers.Flatten(),
#         layers.Dense(20 * 24 * 4, activation=act),

#         layers.Dense(2**n_neurons, activation=act),
#         layers.Dense(units=bottle, activation=act),
#         layers.Dense(2**n_neurons, activation=act),

#         layers.Dense(20 * 24 * 4, activation=act)
#         ])
#     elif nlayers == 2:
#         model = keras.Sequential([
#         layers.Flatten(),
#         layers.Dense(20 * 24 * 4, activation=act),

#         layers.Dense(2**(n_neurons+1), activation=act),
#         layers.Dense(2**n_neurons, activation=act),
#         layers.Dense(units=bottle, activation=act),
#         layers.Dense(2**n_neurons, activation=act),
#         layers.Dense(2**(n_neurons+1), activation=act),

#         layers.Dense(20 * 24 * 4, activation=act)
#         ])
#     else:
#         model = keras.Sequential([
#         layers.Flatten(),
#         layers.Dense(20 * 24 * 4, activation=act),

#         layers.Dense(2**(n_neurons+2), activation=act),
#         layers.Dense(2**(n_neurons+1), activation=act),
#         layers.Dense(2**n_neurons, activation=act),
#         layers.Dense(units=bottle, activation=act),
#         layers.Dense(2**n_neurons, activation=act),
#         layers.Dense(2**(n_neurons+1), activation=act),
#         layers.Dense(2**(n_neurons+2), activation=act),

#         layers.Dense(20 * 24 * 4, activation=act)
#         ])

#     model.compile(optimizer='adam', loss=losses.MeanSquaredError())
#     return model

# tuner = keras_tuner.RandomSearch(
#     build_model,
#     objective='val_loss',
#     max_trials=90)

# tuner.search(training_data_resh, training_data_resh, epochs=200, validation_data=(validation_data_resh, validation_data_resh))
# best_model = tuner.get_best_models()[0]



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

    run = True
    if run:
        autoencoder = Autoencoder()
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        history = autoencoder.fit(training_data_resh, training_data_resh,
                        batch_size=5,  # play with me
                        epochs= 5,  # 200
                        shuffle=True,
                        validation_data=(validation_data_resh, validation_data_resh))

        loss = list(history.history.values())
        autoencoder.encoder.save('encoder')
        autoencoder.decoder.save('decoder')
        encoded = autoencoder.encoder(validation_data_resh).numpy()
        decoded = autoencoder.decoder(encoded).numpy()
    else:
        encoder = tf.keras.models.load_model('encoder')
        decoder = tf.keras.models.load_model('decoder')
        encoded = encoder(validation_data_resh).numpy()
        decoded = decoder(encoded).numpy()

    path = './output/atm_{}/'.format(year)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    np.savetxt('output/atm_{}/encoded.txt'.format(year), encoded, delimiter=',')
    np.savetxt('output/atm_{}/training_data.txt'.format(year), training_data_resh, delimiter=',')
    np.savetxt('output/atm_{}/validation_data.txt'.format(year), validation_data_resh, delimiter=',')
    np.savetxt('output/atm_{}/decoded.txt'.format(year), decoded, delimiter=',')
    if run:
        plt.figure()
        plt.rcParams.update({'font.size': 14})  # increase the font size
        mpl.rcParams['legend.fontsize'] = 15
        plt.xlabel("Number of Epoch")
        plt.ylabel("Loss")
        plt.plot(loss[0], label="Train", linewidth=2)
        plt.plot(loss[1], label="Validation", linewidth=2)
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/loss_atm_{}.png'.format(year))
    error = decoded - validation_data_resh 
    error = np.reshape(error, newshape=(nPoints_val, 20, 24, 4))
    plt.figure() 
    plt.rcParams.update({'font.size': 14})  # increase the font size
    mpl.rcParams['legend.fontsize'] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[7, :19, :, 0])/validation_data[7, :19, :, 0]*100,
                    cmap="inferno", levels=900)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/ReconstructionError_nn_{}.png'.format(year))
    error_norm_nn = linalg.norm(decoded - validation_data_resh)  # , ord=inf
    print('error_norm_nn:', error_norm_nn)
    np.savetxt('output/atm_{}/encoded.txt'.format(year), error_norm_nn, delimiter=',')
