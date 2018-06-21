from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import numpy as np
from sklearn.cluster import SpectralClustering
from data_loader import load_mnist
from pca import pca_plot
from sklearn.cluster import KMeans
import scipy
from functools import partial


def RBF_kernel(x1, x2, gamma):
    return np.exp(-1.0*gamma*np.linalg.norm(x1-x2))


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def new_kernel(x1, x2, gamma):
    return linear_kernel(x1, x2) + RBF_kernel(np.array(x1), np.array(x2), gamma)


def get_affinity(inputs, kernel_fn, gamma=None):
    A_mat = np.zeros((inputs.shape[0], inputs.shape[0]))
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[0]):
            A_mat[i, j] = kernel_fn(inputs[i], inputs[j])
    return A_mat


def normalized_cut_laplacian(affinity_mat):
    """ D**(-1/2) W D**(-1/2)
    """
    D_mat = np.zeros(affinity_mat.shape)
    d_trace = np.sum(affinity_mat, axis=0)
    D_mat.flat[::D_mat.shape[0]+1] = d_trace**(-0.5)
    return D_mat.dot(affinity_mat).dot(D_mat)


def ratio_cut_laplacian(affinity_mat):
    """ L = D - W
    """
    D_mat = np.zeros(affinity_mat.shape)
    d_trace = np.sum(affinity_mat, axis=0)
    D_mat.flat[::D_mat.shape[0]+1] = d_trace
    return D_mat - affinity_mat


def spectral_clustering(inputs, kernel_fn, num_clusters, mode):
    """
    step 1: construct sililarity graph, A as graph's weighted adjacency matrix
    step 2: compute Laplacian Cut
        mode==NCUT: L = D**(-1/2) A D**(-1/2)
        mode==RCUT: L = D - A
    step 3: compute the first k eigenvector of Laplacian Cut, U contains k eigenvectors
        mode==NCUT: get T by normalizing all rows of U to norm 1
        mode==RCUT: U
    setp 4: cluster all points with k-means algorithm in k-dims (k eigenvectors) space.
    """
    # step 1: construct sililarity graph, A as graph's weighted adjacency matrix
    A = get_affinity(inputs, kernel_fn)
    # step 2: compute Laplacian Cut
    if mode == 'NCUT':
        L = normalized_cut_laplacian(A)
    elif mode == 'RCUT':
        L = ratio_cut_laplacian(A)
    # step 3: compute the first k eigenvector of Laplacian Cut, U contains k eigenvectors
    eigen_vals, eigen_vecs = scipy.sparse.linalg.eigs(L, num_clusters)
    if mode == 'NCUT':
        # get T by normalizing all rows of U to norm 1
        rows_norm = np.linalg.norm(eigen_vecs.real, axis=1, ord=2)
        U = (eigen_vecs.real.T/rows_norm).T
    else:
        U = eigen_vecs
    # setp 4: cluster all points with k-means algorithm in k-dims (k eigenvectors) space.
    return KMeans(num_clusters).fit(U).labels_


def sklearn(data):
    # sklearn library support nomalized cut
    normalized_cut = SpectralClustering(
        n_clusters=5, n_jobs=-1, affinity="nearest_neighbors")
    normalized_cut_labels = normalized_cut.fit_predict(
        data['X_train'][:5000:25])
    normalized_cut_labels = normalized_cut_labels + 1
    pca_plot(data['X_train'][:5000:25], data['T_train']
             [:5000:25], with_legend=False)
    pca_plot(data['X_train'][:5000:25],
             normalized_cut_labels, with_legend=False)


def main():
    mode_list = ['RCUT', 'NCUT']
    linear_kernel_partial = partial(linear_kernel)
    RBF_kernel_partial = partial(RBF_kernel, gamma=1.0/28)
    new_kernel_partial = partial(new_kernel, gamma=1.0/28)
    kernel_list = [linear_kernel_partial, RBF_kernel_partial, new_kernel_partial]
    data = load_mnist()
    for mode in mode_list:
        for kernel in kernel_list:
            result = spectral_clustering(
                data['X_train'][:5000:1], kernel_fn=kernel, num_clusters=5, mode=mode)
            # add one, because label are from 1 to 5, cluster index are from 0 to 4
            result = result + 1
            file_name = '{}_{}'.format(mode, kernel.func.__name__)
            pca_plot(data['X_train'][:5000:1],
                     result, with_legend=False, file_name=file_name)


if __name__ == '__main__':
    main()
