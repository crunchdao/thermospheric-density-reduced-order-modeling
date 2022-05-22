import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plot = 0  # 1 true, 0 false
linear = 0  # 1 true, 0 false


def get_lorenz_data(n_ics, noise_strength=0):
    """
        Generate a set of Lorenz training data for multiple random initial conditions.

        Arguments:
            n_ics - Integer specifying the number of initial conditions to use.
            noise_strength - Amount of noise to add to the data.

        Return:
            data - Dictionary containing elements of the dataset. See generate_lorenz_data()
            doc string for list of contents.
        """
    t = np.arange(0, 5, .02)
    input_dim = 128

    ic_means = np.array([0, 0, 25])
    ic_widths = 2 * np.array([36, 48, 41])

    # training data
    ics = ic_widths * (np.random.rand(n_ics, 3) - .5) + ic_means
    data = generate_lorenz_data(ics, t, input_dim, normalization=np.array([1/90, 1/90, 1/90]))
    # data['x'] = data['x'].reshape((-1,input_dim)) + noise_strength*np.random.randn(n_steps*n_ics,input_dim)

    return data


def simulate_lorenz(z0, t, sigma=10., beta=8 / 3, rho=28.):
    """
    Simulate the Lorenz dynamics.

    Arguments:
        z0 - Initial condition in the form of a 3-value list or array.
        t - Array of time points at which to simulate.
        sigma, beta, rho - Lorenz parameters

    Returns:
        z - Array of the trajectory values.
    """
    f = lambda z, t: [sigma * (z[1] - z[0]), z[0] * (rho - z[2]) - z[1], z[0] * z[1] - beta * z[2]]
    z = odeint(f, z0, t)
    return z


def generate_lorenz_data(ics, t, n_points, normalization=None, sigma=10, beta=8/3, rho=28):
    """
    Generate high-dimensional Lorenz data set.

    Arguments:
        ics - Nx3 array of N initial conditions
        t - array of time points over which to simulate
        n_points - size of the high-dimensional dataset created
        linear - Boolean value. If True, high-dimensional dataset is a linear combination
        of the Lorenz dynamics. If False, the dataset also includes cubic modes.
        normalization - Optional 3-value array for rescaling the 3 Lorenz variables.
        sigma, beta, rho - Parameters of the Lorenz dynamics.

    Returns:
        data - Dictionary containing elements of the dataset. This includes the time points (t),
        spatial mapping (y_spatial), high-dimensional modes used to generate the full dataset
        (modes), low-dimensional Lorenz dynamics (z, along with 1st and 2nd derivatives dz and
        ddz), high-dimensional dataset (x, along with 1st and 2nd derivatives dx and ddx), and
        the true Lorenz coefficient matrix for SINDy.
    """
    n_ics = ics.shape[0]
    n_steps = t.size

    d = 3
    z = np.zeros((n_ics, n_steps, d))
    for i in range(n_ics):
        z[i] = simulate_lorenz(ics[i], t, sigma=sigma, beta=beta, rho=rho)

    # test = z.max()

    if normalization is not None:
        z *= normalization

    np.savetxt('output/lorenz%s.txt' % linear, np.reshape(z, (len(z[:, 0, 0])*len(z[0, :, 0]), len(z[0, 0, :]))), delimiter=',')

    n = n_points
    L = 1
    y_spatial = np.linspace(-L, L, n)
    modes = np.zeros((2 * d, n))
    for i in range(2 * d):
        test = legendre(i)
        modes[i] = legendre(i)(y_spatial)

    x1 = np.zeros((n_ics, n_steps, n))
    x2 = np.zeros((n_ics, n_steps, n))
    x3 = np.zeros((n_ics, n_steps, n))
    x4 = np.zeros((n_ics, n_steps, n))
    x5 = np.zeros((n_ics, n_steps, n))
    x6 = np.zeros((n_ics, n_steps, n))

    x = np.zeros((n_ics, n_steps, n))
    for i in range(n_ics):
        for j in range(n_steps):
            x1[i, j] = modes[0]*z[i, j, 0]
            x2[i, j] = modes[1]*z[i, j, 1]
            x3[i, j] = modes[2]*z[i, j, 2]
            x4[i, j] = modes[3]*z[i, j, 0]**3
            x5[i, j] = modes[4]*z[i, j, 1]**3
            x6[i, j] = modes[5]*z[i, j, 2]**3

            if linear:
                x[i, j] = x1[i, j] + x2[i, j] + x3[i, j]
            else:
                x[i, j] = x1[i, j] + x2[i, j] + x3[i, j] + x4[i, j] + x5[i, j] + x6[i, j]

    # data = {}
    # data['t'] = t
    # data['y_spatial'] = y_spatial
    # data['modes'] = modes
    # data['x'] = x
    # data['z'] = z

    return x  # data


def main():

    if plot == 0:
        # generate training, validation, testing data
        noise_strength = 0.0  # 1e-6
        training_data = get_lorenz_data(1024, noise_strength=noise_strength)
        validation_data = get_lorenz_data(10, noise_strength=noise_strength)

        training_data = np.reshape(training_data, (1024*250, 128))
        validation_data = np.reshape(validation_data, (10*250, 128))

        latent_dim = 3

        if linear:
            class Autoencoder(Model):
                def __init__(self, latent_dim):
                    super(Autoencoder, self).__init__()
                    self.latent_dim = latent_dim
                    self.encoder = tf.keras.Sequential([
                        layers.Flatten(),
                        layers.Dense(latent_dim, activation='linear')  # relu, sigmoid
                    ])
                    self.decoder = tf.keras.Sequential([
                        layers.Dense(128, activation='linear')
                    ])

                def call(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
        else:
            class Autoencoder(Model):
                def __init__(self, latent_dim):
                    super(Autoencoder, self).__init__()
                    self.latent_dim = latent_dim
                    self.encoder = tf.keras.Sequential([
                        layers.Flatten(),
                        # layers.Dense(256, activation='relu'),
                        layers.Dense(128, activation='tanh'),
                        layers.Dense(64, activation='tanh'),
                        # layers.Dense(32, activation='relu'),
                        # layers.Dense(16, activation='relu'),
                        layers.Dense(latent_dim, activation='tanh')
                    ])
                    self.decoder = tf.keras.Sequential([
                        # layers.Dense(16, activation='relu'),
                        # layers.Dense(32, activation='relu'),
                        layers.Dense(64, activation='tanh'),
                        layers.Dense(128, activation='tanh'),
                        # layers.Dense(256, activation='relu'),
                        layers.Dense(128, activation='tanh')
                    ])

                def call(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded

        autoencoder = Autoencoder(latent_dim)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        history = autoencoder.fit(training_data, training_data,
                        epochs= 500,  # 20
                        shuffle=True,
                        validation_data=(validation_data, validation_data))

        loss = list(history.history.values())

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
        plt.savefig('output/loss%s_5epochs.png' % linear)


        encoded = autoencoder.encoder(validation_data).numpy()
        decoded = autoencoder.decoder(encoded).numpy()

        np.savetxt('output/encoded%s.txt' % linear, encoded, delimiter=',')
        np.savetxt('output/training_data%s.txt' % linear, training_data, delimiter=',')
        np.savetxt('output/validation_data%s.txt' % linear, validation_data, delimiter=',')
        np.savetxt('output/decoded%s.txt' % linear, decoded, delimiter=',')
    else:
        # Change initial condition in order to obtain nicer plots.

        l = len(np.arange(0, 5, .02))

        lorenz0_df = pd.read_csv('output/lorenz0.txt')
        lorenz0_np = pd.DataFrame.to_numpy(lorenz0_df)

        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot(lorenz0_np[:l, 0], lorenz0_np[:l, 1], lorenz0_np[:l, 2], linewidth=2)
        ax1.set_xlabel(r'$z_1$')
        ax1.set_ylabel(r'$z_2$')
        ax1.set_zlabel(r'$z_3$')

        ax1.view_init(azim=120)
        plt.savefig('output/lorenz0.png', bbox_inches = "tight")

        lorenz1_df = pd.read_csv('output/lorenz1.txt')
        lorenz1_np = pd.DataFrame.to_numpy(lorenz1_df)

        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot(lorenz1_np[2*l:3*l, 0], lorenz1_np[2*l:3*l, 1], lorenz1_np[2*l:3*l, 2], linewidth=2)
        ax1.set_xlabel(r'$z_1$')
        ax1.set_ylabel(r'$z_2$')
        ax1.set_zlabel(r'$z_3$')
        ax1.view_init(azim=120)
        plt.savefig('output/lorenz1.png', bbox_inches = "tight")

        encoded0_df = pd.read_csv('output/encoded0.txt')
        encoded0_np = pd.DataFrame.to_numpy(encoded0_df)

        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot(encoded0_np[:l, 0], encoded0_np[:l, 1], encoded0_np[:l, 2], linewidth=2)
        ax1.set_xlabel(r'$z_1$')
        ax1.set_ylabel(r'$z_2$')
        ax1.set_zlabel(r'$z_3$')
        ax1.view_init(azim=120)
        plt.savefig('output/encoded0.png', bbox_inches = "tight")

        # encoded1_df = pd.read_csv('output/encoded1.txt')
        # encoded1_np = pd.DataFrame.to_numpy(encoded1_df)

        # fig1 = plt.figure(figsize=(3, 3))
        # ax1 = fig1.add_subplot(111, projection='3d')
        # ax1.plot(encoded1_np[2*l:3*l, 0], encoded1_np[2*l:3*l, 1], encoded1_np[2*l:3*l, 2], linewidth=2)
        # ax1.set_xlabel(r'$z_1$')
        # ax1.set_ylabel(r'$z_2$')
        # ax1.set_zlabel(r'$z_3$')
        # plt.show()
        # ax1.view_init(elev=100, azim=180)  # 80 200
        # plt.savefig('output/encoded1.png', bbox_inches = "tight")

        vd0_df = pd.read_csv('output/validation_data0.txt')
        vd0_np = pd.DataFrame.to_numpy(vd0_df)

        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111) # , projection='3d')
        ax1.plot(vd0_np[:l, 0], vd0_np[:l, 1], linewidth=2)  # , vd0_np[:l, 2]
        ax1.set_xlabel(r'$x_1$')
        ax1.set_ylabel(r'$x_2$')
        # ax1.set_zlabel(r'$x_3$')
        plt.grid(True)

        # ax1.view_init(elev=0, azim=90)
        plt.savefig('output/vd0.png', bbox_inches="tight")

        vd1_df = pd.read_csv('output/validation_data1.txt')
        vd1_np = pd.DataFrame.to_numpy(vd1_df)

        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111) # , projection='3d')
        ax1.plot(vd1_np[:l, 0], vd1_np[:l, 1], linewidth=2)  # , vd1_np[:l, 2]
        ax1.set_xlabel(r'$x_1$')
        ax1.set_ylabel(r'$x_2$')
        # ax1.set_zlabel(r'$x_3$')
        plt.grid(True)

        # ax1.view_init(azim=120)
        plt.savefig('output/vd1.png', bbox_inches="tight")

        decoded0_df = pd.read_csv('output/decoded0.txt')
        decoded0_np = pd.DataFrame.to_numpy(decoded0_df)

        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111) # , projection='3d')
        ax1.plot(decoded0_np[:l, 0], decoded0_np[:l, 1], linewidth=2) # , decoded0_np[:l, 2]
        ax1.set_xlabel(r'$x_1$')
        ax1.set_ylabel(r'$x_2$')
        # ax1.set_zlabel(r'$x_3$')
        plt.grid(True)
        # ax1.view_init(azim=120)
        plt.savefig('output/decoded0.png', bbox_inches="tight")

        # decoded1_df = pd.read_csv('output/decoded1.txt')
        # decoded1_np = pd.DataFrame.to_numpy(decoded1_df)

        # fig1 = plt.figure(figsize=(3, 3))
        # ax1 = fig1.add_subplot(111)  #, projection='3d')
        # ax1.plot(decoded1_np[:l, 0], decoded1_np[:l, 1], linewidth=2)  # , decoded1_np[:l, 2]
        # ax1.set_xlabel(r'$x_1$')
        # ax1.set_ylabel(r'$x_2$')
        # # ax1.set_zlabel(r'$x_3$')
        # plt.grid(True)
        # # ax1.view_init(azim=120)
        # plt.savefig('output/decoded1.png', bbox_inches="tight")

        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111) # , projection='3d')
        ax1.plot(vd0_np[:l, 0] - decoded0_np[:l, 0], vd0_np[:l, 1] - decoded0_np[:l, 1], 'r', linewidth=2)  # , vd0_np[:l, 2] - decoded0_np[:l, 2]
        ax1.set_xlabel(r'$x_1$')
        ax1.set_ylabel(r'$x_2$')
        # ax1.set_zlabel(r'$x_3$')
        plt.grid(True)

        # ax1.view_init(azim=120)
        plt.savefig('output/delta0.png', bbox_inches="tight")

        # fig1 = plt.figure(figsize=(3, 3))
        # ax1 = fig1.add_subplot(111)  # , projection='3d')
        # ax1.plot(vd1_np[:l, 0] - decoded1_np[:l, 0], vd1_np[:l, 1] - decoded1_np[:l, 1], 'r', linewidth=2)  #  vd1_np[:l, 2] - decoded1_np[:l, 2],
        # ax1.set_xlabel(r'$x_1$')
        # ax1.set_ylabel(r'$x_2$')
        # # ax1.set_zlabel(r'$x_3$')
        # plt.grid(True)

        # # ax1.view_init(azim=120)
        # plt.savefig('output/delta1.png', bbox_inches="tight")

    return 0


if __name__ == '__main__':
    main()
