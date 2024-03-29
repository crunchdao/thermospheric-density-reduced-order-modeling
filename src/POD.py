from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import differential_evolution, minimize
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import normalize

mpl.use("Agg")


from pyJoules.energy_meter import measure_energy

density_df_500_2013 = pd.read_csv(
    "Data/2013_HASDM_500-575KM.den", delim_whitespace=True, header=None
)
density_np_500_2013 = pd.DataFrame.to_numpy(density_df_500_2013)
del density_df_500_2013

nt = 19
nphi = 24

t = np.linspace(-np.pi / 2, np.pi / 2, nt)
phi = np.linspace(0, np.deg2rad(345), nphi)

max_rho = np.max(density_np_500_2013[:, -1])

density_np_500_2013[:, -1] = density_np_500_2013[:, -1] / max_rho

rho_list = []
rho_list1 = []
rho_list2 = []

for i in range(int(1331520 / (nt * nphi))):  # 1335168
    rho_500_i = density_np_500_2013[i * (4 * nt * nphi) : (i + 1) * (4 * nt * nphi), -1]
    rho_polar_500_i_2013 = np.reshape(rho_500_i, (nt, nphi, 4))

    rho_list1.append(rho_polar_500_i_2013)


rho = np.array(rho_list1)

rho_zeros = np.zeros((2920, 20, 24, 4))
rho_zeros[:, :nt, :nphi, :] = rho
del rho_list1, rho

training_data = rho_zeros[:2000]
validation_data = rho_zeros[2000:]

training_data_resh = np.reshape(training_data, newshape=(2000, 20 * 24 * 4))
nPoints_val = len(rho_zeros) - len(training_data)
validation_data_resh = np.reshape(validation_data, newshape=(nPoints_val, 20 * 24 * 4))


nPoints_val = 920
validation_data_resh = np.reshape(validation_data, newshape=(nPoints_val, 20 * 24 * 4))
rhoavg = np.mean(validation_data_resh, axis=0)  # Compute mean
rho_msub_val = (
    validation_data_resh.T - np.tile(rhoavg, (nPoints_val, 1)).T
)  # Mean-subtracted data

rhoavg = np.mean(training_data_resh, axis=0)  # Compute mean
nPoints = 2000  # 2000
rho_msub = (
    training_data_resh.T - np.tile(rhoavg, (nPoints, 1)).T
)  # Mean-subtracted data
num_modes = 10


@measure_energy
def kpa_script():
    def mykpca(x):
        try:
            kpca = KernelPCA(
                n_components=10,
                kernel="rbf",
                fit_inverse_transform=True,
                gamma=x[0],
                alpha=4.66323895e-12,
            )
            X_pca = kpca.fit_transform(rho_msub.T)
            X_back = kpca.inverse_transform(X_pca)
            X_back = X_back.T
            error_adv = rho_msub - X_back
            error_norm_adv = linalg.norm(error_adv)
            return error_norm_adv
        except:
            return 1e10

    gamma_init = np.linspace(0.1, 0.6, num=5)
    alpha_init = np.linspace(9.52148437e-14, 9.52148437e-9, num=5)
    res_x = []
    res_value = []
    for i in range(1):
        x0 = gamma_init[i]  # alpha_init[i]
        # bounds = [(0.2, 0.4), (1e-14, 1e-12)]
        res = minimize(
            mykpca,
            x0,
            method="Nelder-Mead",
            tol=1e-13,
            options={"maxiter": 10, "disp": True},
        )
        # res = differential_evolution(mykpca, bounds, maxiter=10, tol=1e-14, disp=True)
        res_x.append(res.x)
        res_value.append(res.fun)
    print(res_x)

    x = [0.2575, 4.66323895e-12]
    kpca1 = KernelPCA(
        n_components=num_modes,
        kernel="rbf",
        fit_inverse_transform=True,
        gamma=x[0],
        alpha=x[1],
    )
    X_pca = kpca1.fit_transform(rho_msub.T)
    X_back = kpca1.inverse_transform(X_pca)
    X_back = X_back.T
    error_adv = rho_msub - X_back
    error_norm_adv = linalg.norm(error_adv)
    print(error_norm_adv)


kpa_script()


@measure_energy
def pca_script():
    pca = PCA(n_components=num_modes)
    pca.fit(rho_msub.T)
    X_pca_lin = pca.fit_transform(rho_msub_val.T)
    X_back_lin = pca.inverse_transform(X_pca_lin)
    X_back_lin = X_back_lin.T
    error_lin = rho_msub_val - X_back_lin
    error_norm_lin = linalg.norm(error_lin)
    print(error_norm_lin)


pca_script()
