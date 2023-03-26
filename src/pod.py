import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import LinearConstraint, differential_evolution, minimize
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

mpl.use("Agg")


from pyJoules.energy_meter import measure_energy

density_df = pd.read_csv(
    f"../Data/2013_HASDM_500-575KM.txt", delim_whitespace=True, header=None
)

density_np = pd.DataFrame.to_numpy(density_df)
del density_df

nt = 19
nphi = 24
alt = 4
t = np.linspace(-np.pi / 2, np.pi / 2, nt)
phi = np.linspace(0, np.deg2rad(345), nphi)

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

training_data_resh = np.reshape(training_data, newshape=(nPoints_tra, nt * nphi * alt))


validation_data_resh = np.reshape(
    validation_data_, newshape=(nPoints_val, nt * nphi * alt)
)

normalization_method = "minmax"  # None, minmax, standardscaler

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
elif normalization_method == None:
    training_data_resh_norm = training_data_resh
    validation_data_resh_norm = validation_data_resh

num_modes = 10


def kpca_kpo(x):
    try:
        kpca = KernelPCA(
            n_components=num_modes,
            kernel="rbf",
            fit_inverse_transform=True,
            gamma=x[0],
            alpha=5e-12,
        )
        kpca.fit(training_data_resh_norm)
        X_kpca = kpca.transform(validation_data_resh_norm)
        X_back = kpca.inverse_transform(X_kpca)
        error_adv = X_back - validation_data_resh_norm
        error_norm_adv = linalg.norm(error_adv)
        return error_norm_adv
    except:
        return 1e5


@measure_energy
def kpca_optimization():
    ### Plotting the error vs kernel parameter
    ker_par_analysis = False
    if ker_par_analysis:
        x = np.arange(-7, -1, 0.25)
        x = 10**x
        kpca_error = []  # np.zeros((1, x.shape[0]))
        for i in range(x.shape[0]):
            kpca = KernelPCA(
                n_components=num_modes,
                kernel="rbf",
                fit_inverse_transform=True,
                gamma=x[i],
                alpha=5e-12,
            )
            kpca.fit(training_data_resh_norm)
            X_kpca = kpca.transform(validation_data_resh_norm)
            X_back = kpca.inverse_transform(X_kpca)
            error_adv = X_back - validation_data_resh_norm
            error_norm_adv = linalg.norm(error_adv)
            if normalization_method == "minmax":
                X_back_original = normalizer.inverse_transform(X_back)
                error_original = np.abs(X_back_original - validation_data_resh)
            elif normalization_method == "standardscaler":
                X_back_original = normalizer.inverse_transform(X_back)
                error_original = np.abs(X_back_original - validation_data_resh)
            elif normalization_method == None:
                X_back_original = X_back
                error_original = np.abs(X_back_original - validation_data_resh)
            error_original_kpca_perc = np.sum(
                error_original / validation_data_resh * 100
            ) / np.size(validation_data_resh)
            kpca_error.append(error_original_kpca_perc)

        plt.figure()
        plt.rcParams.update({"font.size": 14})  # increase the font size
        mpl.rcParams["legend.fontsize"] = 15
        plt.xlabel(r"Kernel parameter $\gamma$")
        plt.ylabel("Reconstruction error[%]")
        plt.semilogx(x, kpca_error, linewidth=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/kpca_vs_gamma.png")
    ###
    # The following function is used for kernel parameter optimization
    gamma_init = [1e-5, 1e-4, 1e-3]
    res_x = []
    res_value = []
    lc = LinearConstraint(1, 0, np.inf)

    for x0 in gamma_init:
        res = minimize(
            kpca_kpo,
            x0,
            method="SLSQP",
            tol=1e-10,
            constraints=lc,
            options={"maxiter": 5, "disp": True},
        )
        # bounds = [(0.001, 0.4)]
        # res = differential_evolution(kpca_kpo, bounds, maxiter=10, atol=1e-18, tol=1e-18, strategy='rand2exp', init='sobol', updating='deferred')

        res_x.append(res.x[0])
        res_value.append(res.fun)

    return gamma_init, res_value


gamma_vec, result_value = kpca_optimization()


@measure_energy
def kpca_execution(gamma_vec, result_value):
    kpca1 = KernelPCA(
        n_components=num_modes,
        kernel="rbf",
        fit_inverse_transform=True,
        gamma=gamma_vec[np.argmin(result_value)],
        alpha=5e-12,
    )
    kpca1.fit(training_data_resh_norm)
    X_kpca = kpca1.transform(validation_data_resh_norm)

    X_back = kpca1.inverse_transform(X_kpca)

    if normalization_method == "minmax":
        X_back_original = normalizer.inverse_transform(X_back)
        error_original = np.abs(X_back_original - validation_data_resh)
    elif normalization_method == "standardscaler":
        X_back_original = normalizer.inverse_transform(X_back)
        error_original = np.abs(X_back_original - validation_data_resh)
    elif normalization_method == None:
        X_back_original = X_back
        error_original = np.abs(X_back_original - validation_data_resh)

    error_original_resh = np.reshape(
        error_original, newshape=(nPoints_val, nt, nphi, alt)
    )

    error_norm_kpca = linalg.norm(X_back - validation_data_resh_norm)
    error_original_kpca = linalg.norm(X_back_original - validation_data_resh)
    error_norm_kpca_mse = (
        linalg.norm(X_back - validation_data_resh_norm) ** 2 / X_back.shape[0]
    )
    error_original_kpca_mse = (
        linalg.norm(X_back_original - validation_data_resh) ** 2
        / X_back_original.shape[0]
    )
    error_original_kpca_perc = np.sum(
        error_original / validation_data_resh * 100
    ) / np.size(validation_data_resh)

    # print("Mean percentage error (KPCA):", error_original_kpca_perc)
    # print("error_original_kpca_mse:", error_original_kpca_mse)
    final_error = np.array(
        [
            [
                error_original_kpca_perc,
                error_norm_kpca,
                error_original_kpca,
                error_norm_kpca_mse,
                error_original_kpca_mse,
            ]
        ]
    )
    header = "error_original_kpca_perc, error_norm_kpca, error_original_kpca, error_norm_kpca_mse,  error_original_kpca_mse"
    path1 = f"./output/"
    isExist = os.path.exists(path1)
    if not isExist:
        os.makedirs(path1)

    np.savetxt(
        "output/error_kpca.txt", final_error, delimiter=", ", header=header, comments=""
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
    plt.savefig(f"output/ReconstructionError_kpca.png")


kpca_execution(gamma_vec, result_value)


@measure_energy
def pca_train(pca, x):
    pca.fit(x)
    return pca


@measure_energy
def pca_script():
    pca = PCA(n_components=num_modes)
    # print(training_data_resh_norm.shape) # (2336, 1824)
    pca = pca_train(pca, training_data_resh_norm)

    X_pca_lin = pca.transform(validation_data_resh_norm)
    X_back_lin = pca.inverse_transform(X_pca_lin)
    # X_back_lin = X_back_lin.T

    if normalization_method == "minmax":
        # error_original  = normalizer.inverse_transform(error_lin)
        X_back_lin_original = normalizer.inverse_transform(X_back_lin)
        error_original = np.abs(X_back_lin_original - validation_data_resh)
    elif normalization_method == "standardscaler":
        # error_original  = normalizer.inverse_transform(error_lin)
        X_back_lin_original = normalizer.inverse_transform(X_back_lin)
        error_original = np.abs(X_back_lin_original - validation_data_resh)
    elif normalization_method == None:
        X_back_lin_original = X_back_lin
        error_original = np.abs(X_back_lin_original - validation_data_resh)

    error_original_resh = np.reshape(
        error_original, newshape=(nPoints_val, nt, nphi, alt)
    )

    error_norm_pca = linalg.norm(X_back_lin - validation_data_resh_norm)
    error_original_pca = linalg.norm(X_back_lin_original - validation_data_resh)
    error_norm_pca_mse = (
        linalg.norm(X_back_lin - validation_data_resh_norm) ** 2 / X_back_lin.shape[0]
    )
    error_original_pca_mse = (
        linalg.norm(X_back_lin_original - validation_data_resh) ** 2
        / X_back_lin_original.shape[0]
    )
    error_original_perc = np.sum(error_original / validation_data_resh * 100) / np.size(
        validation_data_resh
    )

    # print("Mean percentage error:", error_original_perc)
    # print("error_norm_pca:", error_norm_pca)
    # print("error_original_pca:", error_original_pca)
    # print("error_norm_pca_mse:", error_norm_pca_mse)
    # print("error_original_pca_mse:", error_original_pca_mse)
    final_error = np.array(
        [
            [
                error_original_perc,
                error_norm_pca,
                error_original_pca,
                error_norm_pca_mse,
                error_original_pca_mse,
            ]
        ]
    )
    header = "error_original_perc, error_norm_pca, error_original_pca, error_norm_pca_mse,  error_original_pca_mse"
    path1 = f"./output/"
    isExist = os.path.exists(path1)
    if not isExist:
        os.makedirs(path1)

    np.savetxt(
        "output/error_pca.txt", final_error, delimiter=", ", header=header, comments=""
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
    plt.savefig("output/ReconstructionError_pca.png")


pca_script()
