import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import modred as mr
from sklearn.decomposition import PCA, KernelPCA
from scipy import linalg


densitypod = 1
if densitypod == 1:
    import matplotlib as mpl
    mpl.use('Agg')

    # density_df_175_2000 = pd.read_csv('../SDC_HASDM/data/2000_HASDM_175-275KM.den', delim_whitespace=True, header=None)
    # density_np_175_2000 = pd.DataFrame.to_numpy(density_df_175_2000)
    # del density_df_175_2000
    #
    # density_df_300_2000 = pd.read_csv('../SDC_HASDM/data/2000_HASDM_300-375KM.den', delim_whitespace=True, header=None)
    # density_np_300_2000 = pd.DataFrame.to_numpy(density_df_300_2000)
    # del density_df_300_2000

    density_df_400_2000 = pd.read_csv('../Data/HASDM/2013_HASDM_400-475KM.den', delim_whitespace=True,
                                      header=None)
    density_np_400_2000 = pd.DataFrame.to_numpy(density_df_400_2000)
    del density_df_400_2000

    # density_df_500_2000 = pd.read_csv('../Data/HASDM/2013_HASDM_500-575KM.den', delim_whitespace=True,
    #                                   header=None)
    # density_np_500_2000 = pd.DataFrame.to_numpy(density_df_500_2000)
    # del density_df_500_2000

    # density_df_600_2000 = pd.read_csv('../SDC_HASDM/data/2000_HASDM_600-675KM.den', delim_whitespace=True,
    #                                   header=None)
    # density_np_600_2000 = pd.DataFrame.to_numpy(density_df_600_2000)
    # del density_df_600_2000
    #
    # density_df_700_2000 = pd.read_csv('../SDC_HASDM/data/2000_HASDM_700-825KM.den', delim_whitespace=True,
    #                                   header=None)
    # density_np_700_2000 = pd.DataFrame.to_numpy(density_df_700_2000)
    # del density_df_700_2000

    nt = 19
    nphi = 24

    t = np.linspace(-np.pi / 2, np.pi / 2, nt)
    phi = np.linspace(0, np.deg2rad(345), nphi)

    # max_rho = np.max(density_np_175_2000[:, 10])  # The maximum is at the lowest altitude, the rest commented out.
    # max2_rho = np.max(density_np_300_2000[:, 10])
    max_rho = np.max(density_np_400_2000[:, 10])
    # max4_rho = np.max(density_np_500_2000[:, 10])
    # max5_rho = np.max(density_np_600_2000[:, 10])
    # max6_rho = np.max(density_np_700_2000[:, 10])

    # max_rho = np.max(np.array([max1_rho, max2_rho, max3_rho, max4_rho, max5_rho, max6_rho]))

    # density_np_175_2000[:, 10] = density_np_175_2000[:, 10] / max_rho
    # density_np_300_2000[:, 10] = density_np_300_2000[:, 10] / max_rho
    density_np_400_2000[:, 10] = density_np_400_2000[:, 10] / max_rho
    # density_np_500_2000[:, 10] = density_np_500_2000[:, 10] / max_rho
    # density_np_600_2000[:, 10] = density_np_600_2000[:, 10] / max_rho
    # density_np_700_2000[:, 10] = density_np_700_2000[:, 10] / max_rho

    rho_list = []
    for i in range(int(1331520 / (nt * nphi))):  # 1335168
        # rho_175_i = density_np_175_2000[i * (5 * nt * nphi):(i + 1) * (5 * nt * nphi), 10]
        # rho_polar_175_i = np.reshape(rho_175_i, (nt, nphi, 5))
        #
        # rho_300_i = density_np_300_2000[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
        # rho_polar_300_i = np.reshape(rho_300_i, (nt, nphi, 4))

        rho_400_i = density_np_400_2000[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
        rho_polar_400_i = np.reshape(rho_400_i, (nt, nphi, 4))

        # rho_500_i = density_np_500_2000[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
        # rho_polar_500_i = np.reshape(rho_500_i, (nt, nphi, 4))

        # rho_600_i = density_np_600_2000[i * (4 * nt * nphi):(i + 1) * (4 * nt * nphi), 10]
        # rho_polar_600_i = np.reshape(rho_600_i, (nt, nphi, 4))
        #
        # rho_700_i = density_np_700_2000[i * (6 * nt * nphi):(i + 1) * (6 * nt * nphi), 10]
        # rho_polar_700_i = np.reshape(rho_700_i, (nt, nphi, 6))

        # rho_polar = np.concatenate(
        #     (rho_polar_400_i, rho_polar_500_i), axis=2)
        rho_polar = rho_polar_400_i
        # rho_polar = np.concatenate(
        #     (rho_polar_175_i, rho_polar_300_i, rho_polar_400_i, rho_polar_500_i, rho_polar_600_i, rho_polar_700_i),
        #     axis=2)

        rho_list.append(rho_polar)
    rho = np.array(rho_list)

    del rho_list

    # del density_np_175_2000
    # del density_np_300_2000
    del density_np_400_2000
    # del density_np_500_2000
    # del density_np_600_2000
    # del density_np_700_2000

    # del rho_polar_175_i
    # del rho_polar_300_i
    del rho_polar_400_i
    # del rho_polar_500_i
    # del rho_polar_600_i
    # del rho_polar_700_i

    rho_zeros = np.zeros((2920, 20, 24, 4))   # 2928        2920, 20, 24, 8
    rho_zeros[:, :nt, :nphi, :] = rho
    del rho

    # rho_zeros = np.expand_dims(rho_zeros, axis=-1)

    training_data = rho_zeros[:2000]
    validation_data = rho_zeros[2000:]
    del rho_zeros

    val_data = validation_data
    # val_data = np.squeeze(validation_data, axis=-1)

    # plt.figure()
    # plt.rcParams.update({'font.size': 14})  # increase the font size
    # mpl.rcParams['legend.fontsize'] = 15
    # plt.xlabel("Longitude [deg]")
    # plt.ylabel("Latitude [deg]")
    # # plt.contourf(np.rad2deg(phi), np.rad2deg(t), val_data[0, :19, :, 0] * max_rho, cmap="viridis", levels=900)
    # plt.contourf(np.rad2deg(phi), np.rad2deg(t), val_data[0, :19, :, 0] * max_rho, cmap="viridis", levels=900)
    # plt.colorbar()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('val_data.png')

    # print(training_data.shape)
    training_data_resh = np.reshape(training_data, newshape=(2000, 20*24*4))
    nPoints_val = 920
    validation_data_resh = np.reshape(validation_data, newshape=(nPoints_val, 20*24*4))
    rhoavg = np.mean(validation_data_resh, axis=0)  # Compute mean
    rho_msub_val = validation_data_resh.T - np.tile(rhoavg, (nPoints_val, 1)).T  # Mean-subtracted data


    # print(training_data_resh.shape)
    rhoavg = np.mean(training_data_resh, axis=0)  # Compute mean
    nPoints = 2000
    rho_msub = training_data_resh.T - np.tile(rhoavg, (nPoints, 1)).T  # Mean-subtracted data
    num_modes = 10
    # print(rho_msub.shape)
    POD_res = mr.compute_POD_arrays_direct_method(
        rho_msub, list(mr.range(num_modes)))
    modes = POD_res.modes
    eigvals = POD_res.eigvals
    # print(np.sqrt(eigvals[:num_modes]))
    ROM = np.matmul(modes.T, rho_msub)
    rho_msub_recon = np.matmul(modes, ROM)
    # energy_perc = np.sum(eigvals[:num_modes])/np.sum(eigvals)
    # print(energy_perc)
    kpca = KernelPCA(n_components=num_modes, kernel="cosine", fit_inverse_transform=True)  # , gamma=10
    # pca = PCA(n_components=num_modes)
    # X_pca = pca.fit_transform(rho_msub.T)
    # pca_val = PCA(n_components=num_modes)
    X_pca = kpca.fit_transform(rho_msub.T)
    # print(X_pca.shape)
    # for i in range(len(num_modes)):
    K = kpca._get_kernel(X_pca)


    class decod(Model):
        def __init__(self, z_size):
            super(decod, self).__init__()
            self.z_size = z_size
            self.decoder = tf.keras.Sequential([
                layers.Input(shape=(z_size)),
                layers.Dense(32, activation='relu'),
                layers.Dense(128, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(20 * 24 * 4, activation='relu')

            ])

        def call(self, x):
            decoded = self.decoder(x)
            return decoded

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

    # exit()
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
    # exit()

    rho_msub_recon_scikit = kpca.inverse_transform(X_pca)
    rho_msub_recon_scikit = rho_msub_recon_scikit.T
    # print(x_recon.shape)
    # print(rho_msub_recon.shape)
    # exit()
    # print(pca.singular_values_)

    k = linalg.lstsq(X_pca, rho_msub.T)
    invtrn = k[0]
    X_back_man = X_pca @ invtrn
    X_back_man = X_back_man.T
    print(rho_msub_recon_scikit[10:15, 500])
    print(X_back_man[10:15, 500])
    print(rho_msub_recon[10:15, 500])
    print(rho_msub[10:15, 500])
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

    error = rho_msub-rho_msub_recon
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
    plt.savefig('ReconstructionError.png')

    error = rho_msub-rho_msub_recon_scikit
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
    plt.savefig('ReconstructionError_scikit.png')

    error = rho_msub-X_back_man
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
    plt.savefig('ReconstructionError_scikit_ls.png')
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

