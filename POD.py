import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, losses
# from tensorflow.keras.models import Model
# import modred as mr
from sklearn.decomposition import PCA, KernelPCA
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy import fftpack
densitypod = 1
if densitypod == 1:
    import matplotlib as mpl
    mpl.use('Agg')

    density_df_400_2013 = pd.read_csv('Data/2013_HASDM_400-475KM.den', delim_whitespace=True,
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

    # nPoints_val = 920  # 840
    # validation_data_resh = np.reshape(validation_data, newshape=(nPoints_val, 20*24*4))
    # rhoavg = np.mean(validation_data_resh, axis=0)  # Compute mean
    # rho_msub_val = validation_data_resh.T - np.tile(rhoavg, (nPoints_val, 1)).T  # Mean-subtracted data

    # print(training_data_resh.shape)
    rhoavg = np.mean(training_data_resh, axis=0)  # Compute mean
    nPoints = 2000
    rho_msub = training_data_resh.T - np.tile(rhoavg, (nPoints, 1)).T  # Mean-subtracted data
    num_modes = 10
    # POD_res = mr.compute_POD_arrays_direct_method(
    #     rho_msub, list(mr.range(num_modes)))
    # modes = POD_res.modes
    # eigvals = POD_res.eigvals
    # ROM = np.matmul(modes.T, rho_msub)
    # rho_msub_recon = np.matmul(modes, ROM)
    # energy_perc = np.sum(eigvals[:num_modes])/np.sum(eigvals)
    # print(energy_perc)

    def cosine_kernel(x, y):
        x_normalized = normalize(x, copy=True)
        if x is y:
            y_normalized = x_normalized
        else:
            y_normalized = normalize(y, copy=True)
        kernels = np.dot(x_normalized, y_normalized.T)
        return kernels


    def my_kernel(X, Y):
        # The best one so far: np.dot(np.sin(X), np.sin(Y).T). It may be better than the linear one if a nonlinear
        # pre-image would exist
        return np.dot(np.sin(X), np.sin(Y).T)


    kpca = KernelPCA(n_components=num_modes, kernel="rbf", fit_inverse_transform=True, gamma=0.11)  # 1.4e-9
    kpca2 = KernelPCA(n_components=num_modes, kernel="sigmoid", fit_inverse_transform=True, gamma=1.4e-2, coef0=1e-5)
    kpca4 = KernelPCA(n_components=num_modes, kernel="laplacian", fit_inverse_transform=True, gamma=1e-2)
    fftout = fftpack.fftn(rho_msub.T, axes=1)

    # pcam = KernelPCA(n_components=num_modes, kernel="precomputed")  # , fit_inverse_transform=True, gamma=10
    pca = PCA(n_components=num_modes)
    pca_fft = PCA(n_components=num_modes)

    X_pca_lin = pca.fit_transform(rho_msub.T)
    X_pca_lin_fft = pca_fft.fit_transform(fftout.real)
    # gram = cosine_kernel(rho_msub.T, rho_msub.T)
    # X_pca_man = pcam.fit_transform(gram)

    X_pca = kpca.fit_transform(rho_msub.T)
    X_pca2 = kpca2.fit_transform(rho_msub.T)
    X_pca4 = kpca4.fit_transform(rho_msub.T)

    auto_recon = False
    if auto_recon:
        print("asd")
        # class decod(Model):
        #         def __init__(self, z_size):
        #         super(decod, self).__init__()
        #         self.z_size = z_size
        #         self.decoder = tf.keras.Sequential([
        #             layers.Input(shape=(z_size)),
        #             layers.Dense(32, activation='relu'),
        #             layers.Dense(128, activation='relu'),
        #             layers.Dense(512, activation='relu'),
        #             layers.Dense(20 * 24 * 4, activation='relu')
        #
        #         ])
        #
        #     def call(self, x):
        #         decoded = self.decoder(x)
        #         return decoded

        # print(X_pca.shape)
        # print(X_pca_val.shape)
        # print(training_data_resh.shape)
        # print(validation_data_resh.shape)
        # exit()
        # mydecoder = decod(num_modes)
        # mydecoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        #
        # history = mydecoder.fit(X_pca, training_data_resh,
        #                       batch_size=5,  # play with me
        #                       epochs=20,  # 25,  # play with me
        #                       shuffle=True,
        #                       validation_data=(X_pca_val, validation_data_resh))
        # mydecoder.decoder.save('Decoder')
        # # loss = list(history.history.values())
        # decoded = mydecoder(X_pca_val).numpy()
        # print(loss)

        # decoder = tf.keras.models.load_model('Decoder')
        # decoded = decoder(X_pca_val).numpy()

    mode_plot = False
    if mode_plot:
        print("asd")
        # plt.figure()
        # # plt.subplot(2, 3, 1)
        # plt.rcParams.update({'font.size': 14})  # increase the font size
        # mpl.rcParams['legend.fontsize'] = 15
        # plt.xlabel("Longitude [deg]")
        # plt.ylabel("Latitude [deg]")
        # # plt.contourf(np.rad2deg(phi), np.rad2deg(t), training_data[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
        # # plt.colorbar()
        # plt.plot(X_pca[:, 0])
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('mode1.png')
        #
        # plt.figure()
        # # plt.subplot(2, 3, 2)
        # plt.rcParams.update({'font.size': 14})  # increase the font size
        # mpl.rcParams['legend.fontsize'] = 15
        # plt.xlabel("Longitude [deg]")
        # plt.ylabel("Latitude [deg]")
        # # plt.contourf(np.rad2deg(phi), np.rad2deg(t), training_data[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
        # # plt.colorbar()
        # plt.plot(X_pca[:, 1])
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('mode2.png')
        #
        # plt.figure()
        # # plt.subplot(2, 3, 3)
        # plt.rcParams.update({'font.size': 14})  # increase the font size
        # mpl.rcParams['legend.fontsize'] = 15
        # plt.xlabel("Longitude [deg]")
        # plt.ylabel("Latitude [deg]")
        # # plt.contourf(np.rad2deg(phi), np.rad2deg(t), training_data[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
        # # plt.colorbar()
        # plt.plot(X_pca[:, 2])
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('mode3.png')
        #
        # plt.figure()
        # # plt.subplot(2, 3, 4)
        # plt.rcParams.update({'font.size': 14})  # increase the font size
        # mpl.rcParams['legend.fontsize'] = 15
        # plt.xlabel("Longitude [deg]")
        # plt.ylabel("Latitude [deg]")
        # # plt.contourf(np.rad2deg(phi), np.rad2deg(t), training_data[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
        # # plt.colorbar()
        # plt.plot(X_pca[:, 3])
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('mode4.png')
        #
        # plt.figure()
        # # plt.subplot(2, 3, 5)
        # plt.rcParams.update({'font.size': 14})  # increase the font size
        # mpl.rcParams['legend.fontsize'] = 15
        # plt.xlabel("Longitude [deg]")
        # plt.ylabel("Latitude [deg]")
        # # plt.contourf(np.rad2deg(phi), np.rad2deg(t), training_data[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
        # # plt.colorbar()
        # plt.plot(X_pca[:, 4])
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('mode5.png')
        # plt.figure()
        # # plt.subplot(2, 3, 6)
        # plt.rcParams.update({'font.size': 14})  # increase the font size
        # mpl.rcParams['legend.fontsize'] = 15
        # plt.xlabel("Longitude [deg]")
        # plt.ylabel("Latitude [deg]")
        # # plt.contourf(np.rad2deg(phi), np.rad2deg(t), training_data[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
        # # plt.colorbar()
        # plt.plot(X_pca[:, 5])
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('mode6.png')

    X_back = kpca.inverse_transform(X_pca)
    X_back2 = kpca2.inverse_transform(X_pca2)
    X_back4 = kpca4.inverse_transform(X_pca4)

    X_back_lin = pca.inverse_transform(X_pca_lin)
    X_back_lin_fft = pca_fft.inverse_transform(X_pca_lin_fft)
    X_back_lin_fft_inv = fftpack.ifftn(X_back_lin_fft, axes=1)

    # K = my_kernel(X_pca,X_pca)
    # K.flat[:: nPoints + 1] += 1
    # dual_coef_ = linalg.solve(K, rho_msub.T, sym_pos=True, overwrite_a=True)
    # K = my_kernel(X_pca, X_pca)
    # X_back_man_nonl = np.dot(K, dual_coef_).T

    X_back_lin = X_back_lin.T
    # X_back1 = X_back1.T
    # X_back_man = X_back_man.T
    X_back = X_back.T
    X_back2 = X_back2.T
    X_back4 = X_back4.T

    error = rho_msub-X_back
    error_norm = linalg.norm(error)
    print('error_norm:', error_norm)  # Error in reconstruction using built-in rbf kpca

    error2 = rho_msub-X_back2
    error_norm2 = linalg.norm(error2)
    print('error_norm2:', error_norm2)
    error4 = rho_msub-X_back4
    error_norm4 = linalg.norm(error4)
    print('error_norm4:', error_norm4)

    # error1 = rho_msub-X_back1
    # error1_norm = linalg.norm(error1)
    # print('error1_norm:', error1_norm)  # Error in reconstruction using built-in cosine kpca with
    # a linear pre-image learning
    # error_man = rho_msub-X_back_man
    # error_norm_man = linalg.norm(error_man)
    # print('error_norm_man:', error_norm_man)  # Error in reconstruction using precomputed cosine kpca with
    # a linear pre-image learning
    error_lin = rho_msub-X_back_lin
    error_norm_lin = linalg.norm(error_lin)  # , ord=inf
    print('error_norm_lin:', error_norm_lin)   # Error in reconstruction using built-in linear pca with
    # error_lin_fft = fftout.real - X_back_lin_fft.T
    error_lin_fft = rho_msub - X_back_lin_fft_inv.T
    error_norm_lin_fft = linalg.norm(error_lin_fft)  # , ord=inf
    print('error_norm_lin_fft:', error_norm_lin_fft)  # Error in reconstruction using built-in linear pca with
    # error_nonl = rho_msub - X_back_man_nonl
    # error_norm_nonl = linalg.norm(error_nonl)
    # print('error_norm_nonl:', error_norm_nonl)  # Error in reconstruction using precomputed cosine kpca with
    # a nonlinear pre-image learning
    # exit()



    # error = -validation_data_resh + decoded
    # error = np.reshape(error, newshape=(920, 20, 24, 4))
    # plt.figure()
    # plt.rcParams.update({'font.size': 14})  # increase the font size
    # mpl.rcParams['legend.fontsize'] = 15
    # plt.xlabel("Longitude [deg]")
    # plt.ylabel("Latitude [deg]")
    # plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[10, :19, :, 0])/validation_data[10, :19, :, 0]*100,
    #              cmap="inferno", levels=900)
    # plt.colorbar()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('ReconstructionError_nn.png')
    # exit()
    # fftin_resh = fftout.real.reshape(2000, 20, 24, 4)
    # fftout_resh = error_lin_fft.reshape(2000, 20, 24, 4)
    error = rho_msub - X_back_lin_fft_inv.T
    error = np.reshape(error.T, newshape=(2000, 20, 24, 4))
    plt.figure()
    plt.rcParams.update({'font.size': 14})  # increase the font size
    mpl.rcParams['legend.fontsize'] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[10, :19, :, 0])/training_data[10, :19, :, 0]*100,
                 cmap="inferno", levels=900)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fft_error.png')

    error = rho_msub-X_back_lin
    error = np.reshape(error.T, newshape=(2000, 20, 24, 4))
    plt.figure()
    plt.rcParams.update({'font.size': 14})  # increase the font size
    mpl.rcParams['legend.fontsize'] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[10, :19, :, 0])/training_data[10, :19, :, 0]*100,
                 cmap="inferno", levels=900)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ReconstructionError_pca.png')

    error = rho_msub-X_back
    error = np.reshape(error.T, newshape=(2000, 20, 24, 4))
    plt.figure()
    plt.rcParams.update({'font.size': 14})  # increase the font size
    mpl.rcParams['legend.fontsize'] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[10, :19, :, 0])/training_data[10, :19, :, 0]*100,
                 cmap="inferno", levels=900)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ReconstructionError_kpca.png')

    # error = rho_msub-X_back_man_nonl
    # error = np.reshape(error.T, newshape=(2000, 20, 24, 4))
    # plt.figure()
    # plt.rcParams.update({'font.size': 14})  # increase the font size
    # mpl.rcParams['legend.fontsize'] = 15
    # plt.xlabel("Longitude [deg]")
    # plt.ylabel("Latitude [deg]")
    # plt.contourf(np.rad2deg(phi), np.rad2deg(t), np.absolute(error[10, :19, :, 0])/training_data[10, :19, :, 0]*100,
    #              cmap="inferno", levels=900)
    # plt.colorbar()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('ReconstructionError_kpca_man.png')
    exit()
    plt.figure()
    plt.rcParams.update({'font.size': 14})  # increase the font size
    mpl.rcParams['legend.fontsize'] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(np.rad2deg(phi), np.rad2deg(t), training_data[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_data.png')

    rho_recon = rho_msub_recon + np.tile(rhoavg, (nPoints, 1)).T
    density_recon = np.reshape(rho_recon.T, newshape=(2000, 20, 24, 4))
    rho_recon_scikit = rho_msub_recon_scikit + np.tile(rhoavg, (nPoints, 1)).T
    density_recon_scikit = np.reshape(rho_recon.T, newshape=(2000, 20, 24, 4))

    plt.figure()
    plt.rcParams.update({'font.size': 14})  # increase the font size
    mpl.rcParams['legend.fontsize'] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(np.rad2deg(phi), np.rad2deg(t), density_recon[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('density_recon.png')

    plt.figure()
    plt.rcParams.update({'font.size': 14})  # increase the font size
    mpl.rcParams['legend.fontsize'] = 15
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.contourf(np.rad2deg(phi), np.rad2deg(t), density_recon_scikit[10, :19, :, 0] * max_rho, cmap="viridis", levels=900)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('density_recon_scikit.png')

if densitypod==0:
    plt.rcParams['figure.figsize'] = [16, 8]

    xC = np.array([2, 1])      # Center of data (mean)
    sig = np.array([2, 0.5])   # Principal axes

    theta = np.pi/3            # Rotate cloud by pi/3

    R = np.array([[np.cos(theta), -np.sin(theta)],     # Rotation matrix
                  [np.sin(theta), np.cos(theta)]])

    nPoints = 10000            # Create 10,000 points
    X = R @ np.diag(sig) @ np.random.randn(2,nPoints) + np.diag(xC) @ np.ones((2,nPoints))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(X[0,:],X[1,:], '.', color='k')
    ax1.grid()
    plt.xlim((-6, 8))
    plt.ylim((-6,8))

    ## f_ch01_ex03_1b

    Xavg = np.mean(X,axis=1)                  # Compute mean
    B = X - np.tile(Xavg,(nPoints,1)).T       # Mean-subtracted data

    # Find principal components (SVD)
    U, S, VT = np.linalg.svd(B,full_matrices=0)

    ax2 = fig.add_subplot(122)
    ax2.plot(X[0, :], X[1, :], '.', color='k')   # Plot data to overlay PCA
    ax2.grid()
    plt.xlim((-6, 8))
    plt.ylim((-6,8))

    theta = 2 * np.pi * np.arange(0, 1, 0.01)

    # 1-std confidence interval
    Xstd = U @ np.diag(S) @ np.array([np.cos(theta),np.sin(theta)])

    ax2.plot(Xavg[0] + Xstd[0,:], Xavg[1] + Xstd[1,:],'-',color='r',linewidth=3)
    ax2.plot(Xavg[0] + 2*Xstd[0,:], Xavg[1] + 2*Xstd[1,:],'-',color='r',linewidth=3)
    ax2.plot(Xavg[0] + 3*Xstd[0,:], Xavg[1] + 3*Xstd[1,:],'-',color='r',linewidth=3)

    # Plot principal components U[:,0]S[0] and U[:,1]S[1]
    ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,0]*S[0]]),
             np.array([Xavg[1], Xavg[1]+U[1,0]*S[0]]),'-',color='cyan',linewidth=5)
    ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,1]*S[1]]),
             np.array([Xavg[1], Xavg[1]+U[1,1]*S[1]]),'-',color='cyan',linewidth=5)

    # plt.show()
    # mr and scikit libraries are validated through Brunton implementation of PCA in the following
    num_modes = 2
    POD_res = mr.compute_POD_arrays_direct_method(
        B, list(mr.range(num_modes)))
    modes = POD_res.modes
    eigvals = POD_res.eigvals
    # rom = POD_res.proj_coeffs
    rom = modes.T @ B
    rec = modes @ rom
    # print(rom[:, 100])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(B.T)
    print(rom[:, 50])
    print(X_pca[50, :])
    # print(U)

    # pca.fit(B.T/np.sqrt(nPoints))

    print(np.sqrt(eigvals[:2]))
    print(pca.singular_values_)
    print(S)

    xrec = pca.inverse_transform(X_pca)
    print(rec[:,100])
    print(xrec[100, :])
    print(B[:, 100])

