import os
from pathlib import Path

import keras_tuner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import fftpack, linalg
from scipy.integrate import odeint
from scipy.special import legendre
from sklearn.preprocessing import normalize
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

# from pyJoules.energy_meter import measure_energy


mpl.use("Agg")

years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"]
years = ["2013"]
for year in years:
    print(year)
    density_df = pd.read_csv(
        f"./Data/{year}_HASDM_500-575KM.txt", delim_whitespace=True, header=None
    )

    density_np = pd.DataFrame.to_numpy(density_df)

    del density_df

    nt = 19
    nphi = 24

    t = np.linspace(-np.pi / 2, np.pi / 2, nt)
    phi = np.linspace(0, np.deg2rad(345), nphi)

    max_rho = np.max(density_np[:, -1])
    density_np[:, -1] = density_np[:, -1] / max_rho

    rho_list = []
    rho_list1 = []
    rho_list2 = []

    for i in range(int(1331520 / (nt * nphi))):
        rho_i = density_np[i * (4 * nt * nphi) : (i + 1) * (4 * nt * nphi), -1]
        rho_polar_i = np.reshape(rho_i, (nt, nphi, 4))

        rho_list1.append(rho_polar_i)

    rho = np.array(rho_list1)

    rho_zeros = np.zeros((2920, 20, 24, 4))
    rho_zeros[:, :nt, :nphi, :] = rho
    del rho_list1, rho

    training_data = rho_zeros[:2000]
    validation_data = rho_zeros[2000:]

    training_data_resh = np.reshape(training_data, newshape=(2000, 20 * 24 * 4))
    nPoints_val = len(rho_zeros) - len(training_data)
    validation_data_resh = np.reshape(
        validation_data, newshape=(nPoints_val, 20 * 24 * 4)
    )

    nPoints_val = 920
    validation_data_resh = np.reshape(
        validation_data, newshape=(nPoints_val, 20 * 24 * 4)
    )
    rhoavg = np.mean(validation_data_resh, axis=0)  # Compute mean
    rho_msub_val = (
        validation_data_resh.T - np.tile(rhoavg, (nPoints_val, 1)).T
    )  # Mean-subtracted data

    rhoavg = np.mean(training_data_resh, axis=0)  # Compute mean
    nPoints = 2000
    rho_msub = (
        training_data_resh.T - np.tile(rhoavg, (nPoints, 1)).T
    )  # Mean-subtracted data

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

    class Autoencoder(Model):
        def __init__(self):
            super().__init__()
            self.encoder = tf.keras.Sequential(
                [
                    layers.Flatten(),
                    layers.Dense(20 * 24 * 4, activation="relu"),
                    layers.Dense(1024, activation="relu"),
                    layers.Dense(512, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(10, activation="relu"),
                ]
            )
            self.decoder = tf.keras.Sequential(
                [
                    layers.Input(shape=(10)),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(512, activation="relu"),
                    layers.Dense(1024, activation="relu"),
                    layers.Dense(20 * 24 * 4, activation="relu"),
                ]
            )

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    run = True
    if run:
        autoencoder = Autoencoder()
        autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

        history = autoencoder.fit(
            training_data_resh,
            training_data_resh,
            batch_size=5,
            epochs=200,
            shuffle=True,
            validation_data=(validation_data_resh, validation_data_resh),
        )

        loss = list(history.history.values())
        autoencoder.encoder.save("encoder")
        autoencoder.decoder.save("decoder")
        encoded = autoencoder.encoder(validation_data_resh).numpy()
        decoded = autoencoder.decoder(encoded).numpy()
    else:
        encoder = tf.keras.models.load_model("encoder")
        decoder = tf.keras.models.load_model("decoder")
        encoded = encoder(validation_data_resh).numpy()
        decoded = decoder(encoded).numpy()

    path = f"./output/atm_{year}/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    np.savetxt(f"output/atm_{year}/encoded.txt", encoded, delimiter=",")
    np.savetxt(
        f"output/atm_{year}/training_data.txt", training_data_resh, delimiter=","
    )
    np.savetxt(
        f"output/atm_{year}/validation_data.txt", validation_data_resh, delimiter=","
    )
    np.savetxt(f"output/atm_{year}/decoded.txt", decoded, delimiter=",")
    if run:
        plt.figure()
        plt.rcParams.update({"font.size": 14})
        mpl.rcParams["legend.fontsize"] = 15
        plt.xlabel("Number of Epoch")
        plt.ylabel("Loss")
        plt.plot(loss[0], label="Train", linewidth=2)
        plt.plot(loss[1], label="Validation", linewidth=2)
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"output/loss_atm_{year}.png")
    error = decoded - validation_data_resh
    error = np.reshape(error, newshape=(nPoints_val, 20, 24, 4))
    plt.figure()
    plt.rcParams.update({"font.size": 14})
    mpl.rcParams["legend.fontsize"] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(
        np.rad2deg(phi),
        np.rad2deg(t),
        np.absolute(error[7, :19, :, 0]) / validation_data[7, :19, :, 0] * 100,
        cmap="inferno",
        levels=900,
    )
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output/ReconstructionError_nn_{year}.png")
    error_norm_nn = linalg.norm(decoded - validation_data_resh)
    print("error_norm_nn:", error_norm_nn)
