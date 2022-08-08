import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, losses
# from tensorflow.keras.models import Model
# import keras_tuner as kt
# import modred as mr
from sklearn.decomposition import PCA, KernelPCA
from scipy import linalg
from sklearn.preprocessing import normalize
# from scipy import fftpack
from pathlib import Path
from scipy.optimize import minimize, differential_evolution

mpl.use('Agg')

# density_df_400_2013 = pd.read_csv('Data/2013_HASDM_400-475KM.den', delim_whitespace=True,
#                                   header=None)
# density_np_400_2013 = pd.DataFrame.to_numpy(density_df_400_2013)
# del density_df_400_2013
# If you want to add more altitudes (increase the initial dimension of the problem), uncomment the followings:
density_df_500_2013 = pd.read_csv('Data/2013_HASDM_500-575KM.den', delim_whitespace=True,
                                  header=None)
density_np_500_2013 = pd.DataFrame.to_numpy(density_df_500_2013)
del density_df_500_2013

# If you want to add more data (increase the time instants), uncomment the followings:
# density_df_400_2014 = pd.read_csv('Data/2014_HASDM_400-475KM.den', delim_whitespace=True,
#                                   header=None)
# density_np_400_2014 = pd.DataFrame.to_numpy(density_df_400_2014)
# del density_df_400_2014

nt = 19
nphi = 24

t = np.linspace(-np.pi / 2, np.pi / 2, nt)
phi = np.linspace(0, np.deg2rad(345), nphi)

# max_rho1 = np.max(density_np_400_2013[:, 10])
# max_rho2 = np.max(density_np_400_2014[:, 10])
# max_rho = np.max(np.array([max_rho1, max_rho2]))
# density_np_400_2013[:, 10] = density_np_400_2013[:, 10] / max_rho
# density_np_400_2014[:, 10] = density_np_400_2014[:, 10] / max_rho

# max_rho = np.max(density_np_400_2013[:, 10])
max_rho = np.max(density_np_500_2013[:, 10])
# max_rho = np.max(np.array([max_rho2, max_rho]))

# density_np_400_2013[:, 10] = density_np_400_2013[:, 10] / max_rho
density_np_500_2013[:, 10] = density_np_500_2013[:, 10] / max_rho
# density_np_400_2014[:, 10] = density_np_400_2014[:, 10] / max_rho

rho_list = []
rho_list1 = []
rho_list2 = []

for i in range(int(1331520 / (nt * nphi))):  # 1335168
    # rho_400_i_2013 = density_np_400_2013[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
    # rho_polar_400_i_2013 = np.reshape(rho_400_i_2013, (nt, nphi, 4))

    # rho_400_i_2014 = density_np_400_2014[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
    # rho_polar_400_i_2014 = np.reshape(rho_400_i_2014, (nt, nphi, 4))

    rho_500_i = density_np_500_2013[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
    rho_polar_500_i_2013 = np.reshape(rho_500_i, (nt, nphi, 4))
    #
    # rho_polar = np.concatenate(
    #     (rho_polar_400_i_2013, rho_polar_400_i_2014), axis=2)

    # rho_polar_2013 = rho_polar_400_i_2013
    # rho_polar_2013 = np.concatenate(
    #     (rho_polar_400_i_2013, rho_polar_500_i_2013), axis=2)

    # rho_polar_2014 = rho_polar_400_i_2014

    # rho_list1.append(rho_polar_400_i_2013)
    # rho_list2.append(rho_polar_400_i_2014)
    rho_list1.append(rho_polar_500_i_2013)



# rho1 = np.array(rho_list1)
# rho2 = np.array(rho_list2)
# rho = np.concatenate((rho1, rho2), axis=0)  # Shape: (5840, 19, 24, 4)

rho = np.array(rho_list1)

rho_zeros = np.zeros((2920, 20, 24, 4))  # 2920, 20, 24, 4 
rho_zeros[:, :nt, :nphi, :] = rho
del rho_list1, rho  #, rho_list2, rho1, rho2

training_data = rho_zeros[:2000] # 2000
validation_data = rho_zeros[2000:]

training_data_resh = np.reshape(training_data, newshape=(2000, 20*24*4))
nPoints_val = len(rho_zeros) - len(training_data)
validation_data_resh = np.reshape(validation_data, newshape=(nPoints_val, 20*24*4))


nPoints_val = 920  # 840 920
validation_data_resh = np.reshape(validation_data, newshape=(nPoints_val, 20*24*4))
rhoavg_val = np.mean(validation_data_resh, axis=0)  # Compute mean
rho_msub_val = validation_data_resh.T - np.tile(rhoavg_val, (nPoints_val, 1)).T  # Mean-subtracted data

# print(training_data_resh.shape)
rhoavg = np.mean(training_data_resh, axis=0)  # Compute mean
nPoints = 2000 # 2000
rho_msub = training_data_resh.T - np.tile(rhoavg, (nPoints, 1)).T  # Mean-subtracted data
num_modes = 10

# def mykpca(x):
#     try: 
#         kpca = KernelPCA(n_components=10, kernel="rbf", fit_inverse_transform=True, gamma=x[0], alpha=4.66323895e-12)
#         X_pca = kpca.fit_transform(rho_msub.T)
#         X_back = kpca.inverse_transform(X_pca)
#         X_back = X_back.T
#         error_adv = rho_msub-X_back
#         error_norm_adv = linalg.norm(error_adv)
#         return error_norm_adv
#     except:
#         return 1e+10
# gamma_init = np.linspace(0.1,0.6,num=5)
# alpha_init = np.linspace(9.52148437e-14, 9.52148437e-9, num=5)
# res_x = []
# res_value = []
# for i in range(1):
#     x0 = gamma_init[i] # alpha_init[i]
#     # bounds = [(0.2, 0.4), (1e-14, 1e-12)]
#     res = minimize(mykpca, x0, method='Nelder-Mead', tol=1e-13, options={'maxiter':10, 'disp': True})
#     # res = differential_evolution(mykpca, bounds, maxiter=10, tol=1e-14, disp=True)
#     res_x.append(res.x)
#     res_value.append(res.fun)
# print(res_x)

x = [0.2575, 4.66323895e-12] # 3.27369962e-01 4.66323895e-14
kpca1 = KernelPCA(n_components=num_modes, kernel="rbf", fit_inverse_transform=True, gamma=x[0], alpha=x[1])
kpca1.fit(rho_msub.T)
X_pca = kpca1.fit_transform(rho_msub_val.T)
X_back = kpca1.inverse_transform(X_pca)
X_back = X_back.T
error_adv = rho_msub_val-X_back
error_norm_adv = linalg.norm(error_adv)
print(error_norm_adv)
error_adv = np.reshape(error_adv, newshape=(nPoints_val, 20, 24, 4))
X_back = X_back + np.tile(rhoavg_val, (nPoints_val, 1)).T
X_back_kpca = np.reshape(X_back, newshape=(nPoints_val, 20, 24, 4))


plt.figure() 
plt.rcParams.update({'font.size': 14})  # increase the font size
mpl.rcParams['legend.fontsize'] = 15
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error_adv[0, :19, :, 0])/validation_data[0, :19, :, 0]*100,
                cmap="inferno", levels=900)
plt.colorbar()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/ReconstructionError_kpca.png')

plt.figure() 
plt.rcParams.update({'font.size': 14})  # increase the font size
mpl.rcParams['legend.fontsize'] = 15
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.contourf(np.rad2deg(phi), np.rad2deg(t), validation_data[0, :19, :, 0]*max_rho,
                cmap="inferno", levels=900)
plt.colorbar()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/dens_input.png')
plt.figure() 
plt.rcParams.update({'font.size': 14})  # increase the font size
mpl.rcParams['legend.fontsize'] = 15
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.contourf(np.rad2deg(phi), np.rad2deg(t), X_back_kpca[0, :19, :, 0]*max_rho,
                cmap="inferno", levels=900)
plt.colorbar()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/recons_kpca.png')

pca = PCA(n_components=num_modes)
pca.fit(rho_msub.T)
X_pca_lin = pca.fit_transform(rho_msub_val.T)
X_back_lin = pca.inverse_transform(X_pca_lin)
X_back_lin = X_back_lin.T
error_lin = rho_msub_val-X_back_lin
error_lin = np.reshape(error_lin, newshape=(nPoints_val, 20, 24, 4))

error_norm_lin = linalg.norm(error_lin)
print(error_norm_lin)
plt.figure() 
plt.rcParams.update({'font.size': 14})  # increase the font size
mpl.rcParams['legend.fontsize'] = 15
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error_lin[0, :19, :, 0])/validation_data[0, :19, :, 0]*100,
                cmap="inferno", levels=900)
plt.colorbar()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/ReconstructionError_pca.png')

# x = [3.27369962e-01, 4.66323895e-12] # 4.66323895e-14
# num_modes = 1
# modes = []
# PCA_error = []
# KPCA_error = []

# for num_modes in range(2,15):
#     modes.append(num_modes)
#     kpca_adv = KernelPCA(n_components=num_modes, kernel="rbf", fit_inverse_transform=True, gamma=x[0], alpha=x[1])

#     # fftout = fftpack.fftn(rho_msub.T)

#     pca = PCA(n_components=num_modes)

#     # pca_fft = PCA(n_components=num_modes)
#     # pca.fit(rho_msub.T)
#     # X_pca_lin = pca.fit_transform(rho_msub_val.T)
#     X_pca_lin = pca.fit_transform(rho_msub.T)

#     # X_pca_lin_fft = pca_fft.fit_transform(fftout.real)

#     # gram = cosine_kernel(rho_msub.T, rho_msub.T)
#     # X_pca_man = pcam.fit_transform(gram)
#     # X_pca = kpca.fit_transform(rho_msub.T)
#     # explained_variance_kpca = (kpca.eigenvalues_ ** 2) / (len(rho_msub.T) - 1)

#     # print(np.sum(explained_variance_kpca))

#     # kpca_test = KernelPCA(n_components=1920, kernel="rbf", fit_inverse_transform=True, gamma=0.11)  #  0.11

#     # kpca_test.fit(rho_msub.T)
#     # X_pca_test = kpca_test.fit_transform(rho_msub_val.T)
#     # X_pca_test = kpca_test.fit_transform(rho_msub.T)
#     # explained_variance_kpca_test = (kpca_test.eigenvalues_ ** 2) / (len(rho_msub.T) - 1)

#     # print(np.sum(explained_variance_kpca)/np.sum(explained_variance_kpca_test))

#     # kpca_adv.fit(rho_msub.T)
#     # X_pca_adv = kpca_adv.fit_transform(rho_msub_val.T)
#     X_pca_adv = kpca_adv.fit_transform(rho_msub.T)

#     X_back_adv = kpca_adv.inverse_transform(X_pca_adv)

#     X_back_lin = pca.inverse_transform(X_pca_lin)
#     # X_back_lin_fft = pca_fft.inverse_transform(X_pca_lin_fft)
#     # X_back_lin_fft_inv = fftpack.ifftn(X_back_lin_fft)

#     # K = my_kernel(X_pca,X_pca)
#     # K.flat[:: nPoints + 1] += 1
#     # dual_coef_ = linalg.solve(K, rho_msub.T, sym_pos=True, overwrite_a=True)
#     # K = my_kernel(X_pca, X_pca)
#     # X_back_man_nonl = np.dot(K, dual_coef_).T

#     X_back_lin = X_back_lin.T

#     # X_back = X_back.T
#     X_back_adv = X_back_adv.T

#     # X_back_lin_fft_inv = X_back_lin_fft_inv.T

#     # error = rho_msub-X_back
#     # error_norm = linalg.norm(error)
#     # print('error_norm_rbf:', error_norm)  # Error in reconstruction using built-in rbf kpca

#     error_adv = rho_msub-X_back_adv
#     error_norm_adv = linalg.norm(error_adv)
#     # print('error_norm_rbf with input alpha:', error_norm_adv)  # Error in reconstruction using built-in rbf kpca
#     KPCA_error.append(error_norm_adv) 


#     error_lin = rho_msub-X_back_lin
#     error_norm_lin = linalg.norm(error_lin)  # , ord=inf
#     # print('error_norm_lin:', error_norm_lin)   # Error in reconstruction using built-in linear pca with
#     PCA_error.append(error_norm_lin) 

#     # error_lin_fft = fftout.real - X_back_lin_fft.T
#     # error_lin_fft = rho_msub - X_back_lin_fft_inv
#     # error_norm_lin_fft = linalg.norm(error_lin_fft)  # , ord=inf
#     # print('error_norm_lin_fft:', error_norm_lin_fft)  # Error in reconstruction using built-in linear pca with


#     # fftin_resh = fftout.real.reshape(2000, 20, 24, 4)
#     # fftout_resh = error_lin_fft.reshape(2000, 20, 24, 4)
#     # error = rho_msub - X_back_lin_fft_inv
#     # error = np.reshape(error.T, newshape=(2000, 20, 24, 4))
#     # Path("./output/").mkdir(parents=True, exist_ok=True)
#     # plt.figure()
#     # plt.rcParams.update({'font.size': 14})  # increase the font size
#     # mpl.rcParams['legend.fontsize'] = 15
#     # plt.xlabel("Longitude [deg]")
#     # plt.ylabel("Latitude [deg]")
#     # plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[10, :19, :, 0])/training_data[10, :19, :, 0]*100,
#     #              cmap="inferno", levels=900)
#     # plt.colorbar()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # plt.savefig('output/fft_error.png')

#     # error = rho_msub-X_back_lin
#     # error = np.reshape(error.T, newshape=(2000, 20, 24, 4))
#     # plt.figure()
#     # plt.rcParams.update({'font.size': 14})  # increase the font size
#     # mpl.rcParams['legend.fontsize'] = 15
#     # plt.xlabel("Longitude [deg]")
#     # plt.ylabel("Latitude [deg]")
#     # plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[10, :19, :, 0])/training_data[10, :19, :, 0]*100,
#     #              cmap="inferno", levels=900)
#     # plt.colorbar()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # plt.savefig('output/ReconstructionError_pca.png')

#     # error_adv = rho_msub-X_back_adv
#     # error_adv = np.reshape(error_adv.T, newshape=(2000, 20, 24, 4))
#     # plt.figure()
#     # plt.rcParams.update({'font.size': 14})  # increase the font size
#     # mpl.rcParams['legend.fontsize'] = 15
#     # plt.xlabel("Longitude [deg]")
#     # plt.ylabel("Latitude [deg]")
#     # plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error_adv[10, :19, :, 0])/training_data[10, :19, :, 0]*100,
#     #              cmap="inferno", levels=900)
#     # plt.colorbar()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # plt.savefig('output/ReconstructionError_kpca_adv.png')

#     # plt.close('all')

# # data = {'number of modes':modes,'PCA error':PCA_error, 'KPCA error':KPCA_error}
# # df = pd.DataFrame(data)
# # print(df)

# errorae = [66.8464, 28.1242, 25.1675, 26.8981, 21.2226, 18.3057]
# modes = [2, 4, 6, 8, 10, 12]
# PCA_error = [16.714446, 9.173239, 6.108573, 3.120358, 1.229338, 0.848673]
# KPCA_error = [25.821876, 8.507736, 4.921118, 2.429778, 0.286163, 0.072301]
# datam = {'number of modes':modes,'PCA error':PCA_error, 'KPCA error':KPCA_error, 'Autoencoder erros':errorae}
# df = pd.DataFrame(datam)
# print(df)
# plt.figure()
# plt.rcParams.update({'font.size': 14})  # increase the font size
# mpl.rcParams['legend.fontsize'] = 15
# plt.xlabel("# of modes")
# plt.ylabel("Reconstruction error")
# plt.plot(PCA_error, label="PCA error", linewidth=2)
# plt.plot(KPCA_error, label="KPCA error", linewidth=2)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('output/PCA_vs_KPCA_error.png')
# plt.close('all')
