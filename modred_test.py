import numpy as np
from scipy import linalg
import modred as mr
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import chi2_kernel
from scipy.optimize import minimize
from sklearn.preprocessing import normalize


# Create random data
num_vecs = 30
nPoints = 100
vecs = np.random.random((nPoints, num_vecs))
vecsavg = np.mean(vecs, axis=1)  # Compute mean
B = vecs - np.tile(vecsavg, (num_vecs, 1)).T  # Mean-subtracted data
# Compute POD
num_modes = 10
POD_res = mr.compute_POD_arrays_direct_method(
    B.T, list(mr.range(num_modes)))
modes = POD_res.modes
eigvals = POD_res.eigvals
# print(modes.shape)

ROM = modes.T @ B.T


def linear_kernel(X, Y):
    return np.dot(X, Y.T)


def two_norm(x):
    return np.linalg.norm(x, ord=2)


def cosine_kernel(x, y):
    # kernels = safe_sparse_dot(normalize(x), normalize(y).T, dense_output=True)
    kernels = np.dot(normalize(x), normalize(y).T)
    return kernels


pca = KernelPCA(n_components=num_modes, fit_inverse_transform=True, kernel="cosine")
X_pca = pca.fit_transform(B)

pcam = KernelPCA(n_components=num_modes, kernel="precomputed")  # , fit_inverse_transform=True, gamma=10
gram = cosine_kernel(B, B)
X_pcam = pcam.fit_transform(gram)

print(X_pca[50, :])
print(X_pcam[50, :])
exit()
k = linalg.lstsq(X_pca, B)
invtrn = k[0]
X_back_man = X_pca @ invtrn
# print(pca.dual_coef_.shape)
# print(pca._get_kernel(X_pca))
# K = linear_kernel(X_pca, X_pca)
# K.flat[:: nPoints + 1] += 1
# dual_coef_ = linalg.solve(K, B, sym_pos=True, overwrite_a=True)
# print(dual_coef_.shape)
# K1 = pca._get_kernel(X_pca, B)
# invtrn = np.dot(K1, dual_coef_)
# print(invtrn.shape)
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# a.flat[::3] +=1
# print(a)
# print(linear_kernel(X_pca,X_pca)[80,80])
# print(K.flat[:: 100 + 1].shape)
X_back = pca.inverse_transform(X_pca)
print(X_back[80, :5])
print(X_back_man[80, :5])
print(B[80, :5])
# print(X_pca[80, :])
# print(pca.X_transformed_fit_[80, :])
# kpcapre = KernelPCA(n_components=num_modes, kernel="precomputed", gamma=0.5)
# gram = linear_kernel(B, B)
# # u,s,vh  = linalg.svd(gram)
# X_kpca1 = kpcapre.fit_transform(gram)
# dual_coef = linalg.solve(gram, B, sym_pos=True, overwrite_a=True)
# print(dual_coef.shape)
# ngram = linear_kernel(B, X_kpca1)
# invtrn = ngram @ dual_coef
# print(invtrn)
# exit()

