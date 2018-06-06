from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from data_loader import load_mnist
import matplotlib.pyplot as plt

colors = ['b', 'c', 'y', 'm', 'r']


def pca_fit_n_transform(inputs, num_components=2):
    """
    Args
    ----
    inputs : float, shape=[num_data, num_dims]

    1. Computing the d-dimensional mean vector
    2. Computing the Covariance Matrix 
    3. Computing eigenvectors and corresponding eigenvalues
    4. Sorting the eigenvectors by decreasing eigenvalues
    5. Choosing k eigenvectors with the largest eigenvalues
    6. Transforming the samples onto the new subspace
    """
    num_data = inputs.shape[0]
    num_dims = inputs.shape[1]
    # 1. Computing the d-dimensional mean vector
    mean_vector = np.mean(inputs, axis=0)
    # 2. Computing the Covariance Matrix
    cov_mat = np.cov([inputs[:, i] for i in range(num_dims)])
    # 3. Computing eigenvectors and corresponding eigenvalues
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    # 4. Sorting the eigenvectors by decreasing eigenvalues
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i])
                 for i in range(num_dims)]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # 5. Choosing k eigenvectors with the largest eigenvalues
    matrix_w = np.stack([eig_pairs[i][1]
                         for i in range(num_components)], axis=1)
    # 6. Transforming the samples onto the new subspace
    return np.dot(inputs, matrix_w)


def pca_plot(inputs, labels, with_legend=True, num_components=2, file_name='default'):
    # PCA
    pca_data = pca_fit_n_transform(inputs, num_components=num_components)
    # Visualization
    fig, ax = plt.subplots()
    scatter_obj = []
    for i in range(5):
        indecies = (labels == (i+1))
        scatter_obj.append(ax.scatter(
            pca_data[indecies, 0], pca_data[indecies, 1], c=colors[i], s=8.0, alpha=0.5))
    if with_legend:
        ax.legend(tuple(scatter_obj),
                  ('1', '2', '3', '4', '5'),
                  scatterpoints=1,
                  loc='lower left',
                  fontsize=8)
    plt.savefig('{}.png'.format(file_name))
    print('Sucessfully Save the Fig: {}'.format(file_name))


def main():
    data = load_mnist()
    pca_plot(data['X_train'], data['T_train'])


if __name__ == '__main__':
    main()
