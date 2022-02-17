import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
from scipy.optimize import minimize
import modred as mr

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=0.3, noise=0.05)

# def linear_kernel(X, Y):
#     return X.dot(Y.T)
#
#
# kpcapre = KernelPCA(kernel="precomputed", gamma=10)
# gram = linear_kernel(X, X)
# # u,s,vh  = linalg.svd(gram)
# X_kpca1 = kpcapre.fit_transform(gram)

# dual_coef = linalg.solve(gram, X, sym_pos=True, overwrite_a=True)
# print(dual_coef.shape)
# exit()
# kernels = linear_kernel(X, X_kpca1)

# asdf = kernels @ dual_coef
# print(asdf[150, :])
# print(X[150, :])
# exit()
K = euclidean_distances(X, X, squared=True)
gamma = 0.5
K *= -gamma
kernel = np.exp(K,K)
pca1 = PCA()
X_pca1 = pca1.fit_transform(kernel)
# print(X_pca1.shape)
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
# print(X_kpca[150, :])
# exit()
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)
# print(X_pca.shape)


# n = 20
# m = 100
# A = np.random.rand(n, m)
# b = np.random.rand(n)


def two_norm(x):
    return np.linalg.norm(x,ord=2)


constr = ({'type': 'eq', 'fun': lambda x:  X_pca @ x - X_back})
x0 = np.random.rand(len(X_back),2)
print(X_back.shape)
print(x0.shape)
print(X_pca.shape)
res = minimize(two_norm, x0, method='SLSQP',constraints=constr)
x2 = res.x
print(x2.shape)
print(x0.shape)
exit()
# print(X_pca[150, :])
# print(X[150,:])
# print(X_back[150,:])
# exit()
# exit()
# POD_res = mr.compute_POD_arrays_direct_method(rho_msub, list(mr.range(num_modes)))
# modes = POD_res.modes
# eigvals = POD_res.eigvals
# print(eigvals[:10])
# exit()
# Plot results

plt.figure()
plt.subplot(2, 2, 1, aspect="equal")
plt.title("Original space")
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor="k")
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor="k")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# projection on the first principal component (in the phi space)
Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
plt.contour(X1, X2, Z_grid, colors="grey", linewidths=1, origin="lower")

plt.subplot(2, 2, 2, aspect="equal")
plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red", s=20, edgecolor="k")
plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue", s=20, edgecolor="k")
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.subplot(2, 2, 3, aspect="equal")
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red", s=20, edgecolor="k")
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue", s=20, edgecolor="k")
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.subplot(2, 2, 4, aspect="equal")
plt.scatter(X_back[reds, 0], X_back[reds, 1], c="red", s=20, edgecolor="k")
plt.scatter(X_back[blues, 0], X_back[blues, 1], c="blue", s=20, edgecolor="k")
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.tight_layout()
plt.show()

