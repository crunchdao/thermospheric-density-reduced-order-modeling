import os
from pathlib import Path

# import keras_tuner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pyJoules.energy_meter import measure_energy
from scipy import fftpack, linalg
from scipy.integrate import odeint
from scipy.special import legendre
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

# X = np.array([[1,2,3],[4,5,6],[7,8,9]])
# np.random.shuffle(X)
# print(X)
# exit()
# normalizer = Normalizer(norm='max')
# print(np.std(X))
# print(np.mean(X))
# asd = (X - np.mean(X))/ np.std(X)
# X1 = normalizer.fit_transform(X)
# print(X1)
# print(asd)
# exit()
mpl.use("Agg")

# years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"]
years = ["2013"]
for year in years:
    print(year)
    density_df = pd.read_csv(
        f"../Data/{year}_HASDM_500-575KM.txt", delim_whitespace=True, header=None
    )

    density_np = pd.DataFrame.to_numpy(density_df)

    del density_df

    nt = 19
    nphi = 24
    alt = 4
    t = np.linspace(-np.pi / 2, np.pi / 2, nt)
    phi = np.linspace(0, np.deg2rad(345), nphi)
    # print(density_np.shape)
    # exit()
    # max_rho = np.max(density_np[:, -1])
    # density_np[:, -1] = density_np[:, -1] / max_rho

    rho_list = []
    rho_list1 = []
    rho_list2 = []

    for i in range(int(1331520 / (nt * nphi))):
        rho_i = density_np[i * (4 * nt * nphi) : (i + 1) * (4 * nt * nphi), -1]

        rho_polar_i = np.reshape(rho_i, (nt, nphi, 4))

        rho_list1.append(rho_polar_i)

    rho = np.array(rho_list1)

    rho_zeros = rho

    del rho_list1, rho

    rho_zeros_shuffled = np.copy(rho_zeros)
    np.random.shuffle(rho_zeros_shuffled)

    nPoints_val = int(0.2 * rho_zeros_shuffled.shape[0])
    nPoints_tra = rho_zeros_shuffled.shape[0] - nPoints_val

    training_data = rho_zeros_shuffled[nPoints_val:]
    validation_data_ = rho_zeros_shuffled[:nPoints_val]

    training_data_resh = np.reshape(
        training_data, newshape=(nPoints_tra, nt * nphi * alt)
    )

    # nPoints_val = len(rho_zeros) - len(training_data)
    # validation_data_resh = np.reshape(
    #     validation_data, newshape=(nPoints_val, 20 * 24 * 4))

    # nPoints_val = 920

    validation_data_resh = np.reshape(
        validation_data_, newshape=(nPoints_val, nt * nphi * alt)
    )

    normalization_method = "minmax"  # minmax, standardscaler

    if normalization_method == "minmax":
        # training_data_resh_norm = [(training_data_resh[:,i]- np.min(training_data_resh[:,i]))/(np.max(training_data_resh[:,i])-np.min(training_data_resh[:,i])) for i in range(training_data_resh.shape[1])]
        # validation_data_resh_norm = [(validation_data_resh[:,i]- np.min(training_data_resh[:,i]))/(np.max(training_data_resh[:,i])-np.min(training_data_resh[:,i])) for i in range(validation_data_resh.shape[1])]
        normalizer = MinMaxScaler()
        training_data_resh_norm = normalizer.fit_transform(training_data_resh)
        validation_data_resh_norm = normalizer.transform(validation_data_resh)
    elif normalization_method == "standardscaler":
        normalizer = StandardScaler()
        training_data_resh_norm = normalizer.fit_transform(training_data_resh)
        validation_data_resh_norm = normalizer.transform(validation_data_resh)
    # print(training_data_resh[10,:3])
    # print(validation_data_resh[10,:3])
    # print(training_data_resh_norm[10,:3])
    # print(validation_data_resh_norm[10,:3])
    # exit()
    # elif normalization_method == 'standardscaler':
    # exit()
    # print(np.max(training_data))
    # print(np.min(training_data))
    # exit()
    # rhoavg = np.mean(validation_data_resh, axis=0)  # Compute mean
    # rho_msub_val = (
    #     validation_data_resh.T - np.tile(rhoavg, (nPoints_val, 1)).T
    # )  # Mean-subtracted data

    # rhoavg = np.mean(training_data_resh, axis=0)  # Compute mean
    # nPoints = 2000
    # rho_msub = (
    #     training_data_resh.T - np.tile(rhoavg, (nPoints, 1)).T
    # )  # Mean-subtracted data

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
                    layers.Dense(nt * nphi * alt, activation="relu"),
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
                    layers.Dense(nt * nphi * alt, activation="relu"),
                ]
            )

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    run = False
    if run:
        autoencoder = Autoencoder()
        autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

        @measure_energy
        def train_script():
            history = autoencoder.fit(
                training_data_resh_norm,
                training_data_resh_norm,
                batch_size=5,
                epochs=20,  # 200
                shuffle=True,
                validation_data=(validation_data_resh_norm, validation_data_resh_norm),
            )
            return history

        history = train_script()

        loss = list(history.history.values())
        autoencoder.encoder.save("encoder")
        autoencoder.decoder.save("decoder")
        encoded = autoencoder.encoder(validation_data_resh_norm).numpy()
        decoded = autoencoder.decoder(encoded).numpy()
    else:
        encoder = tf.keras.models.load_model("encoder")
        decoder = tf.keras.models.load_model("decoder")

        import time

        # Grab Currrent Time Before Running the Code
        start = time.time()

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
    error = decoded - validation_data_resh_norm
    if normalization_method == "minmax":
        error_original = normalizer.inverse_transform(error)
        decoded_original = normalizer.inverse_transform(decoded)
    elif normalization_method == "standardscaler":
        error_original = normalizer.inverse_transform(error)
        decoded_original = normalizer.inverse_transform(decoded)
    error_original_resh = np.reshape(
        error_original, newshape=(nPoints_val, nt, nphi, alt)
    )

    plt.figure()
    plt.rcParams.update({"font.size": 14})
    mpl.rcParams["legend.fontsize"] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(
        np.rad2deg(phi),
        np.rad2deg(t),
        np.absolute(error_original_resh[7, :19, :, 0])
        / validation_data_[7, :19, :, 0]
        * 100,
        cmap="inferno",
        levels=900,
    )
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output/ReconstructionError_nn_{year}.png")
    error_norm_nn = linalg.norm(decoded - validation_data_resh_norm)
    error_original_nn = linalg.norm(decoded_original - validation_data_resh)
    error_norm_nn_mse = (
        linalg.norm(decoded - validation_data_resh_norm) ** 2 / decoded.shape[0]
    )
    error_original_nn_mse = (
        linalg.norm(decoded_original - validation_data_resh) ** 2 / decoded.shape[0]
    )
    error_original_perc = np.sum(error_original / validation_data_resh * 100) / np.size(
        validation_data_resh
    )
    # Grab Current Time After Running the Code
    end = time.time()

    # Subtract Start Time from The End Time
    total_time = end - start
    print("\n" + str(total_time))

    print("error_norm_nn:", error_norm_nn)
    print("error_original_nn:", error_original_nn)
    print("error_norm_nn:", error_norm_nn_mse)
    print("error_original_nn:", error_original_nn_mse)
    print("Mean percentage error:", error_original_perc)
final_error = [
    error_original_perc,
    error_norm_nn,
    error_original_nn,
    error_norm_nn_mse,
    error_original_nn_mse,
]
path1 = f"./output/"
isExist = os.path.exists(path1)
if not isExist:
    os.makedirs(path1)

np.savetxt(f"output/error_ld.txt", final_error, delimiter=",")
