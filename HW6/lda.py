from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from data_loader import load_mnist
import matplotlib.pyplot as plt

colors = ['b', 'c', 'y', 'm', 'r']


def lda_fit_n_transform(inputs, labels, num_components):
    """
    1. Calculate means for each class
    2. Compute the Covariance Matrix between classes
    3. Compute the Covariance Matrix within classes
    4. Computing eigenvectors and corresponding eigenvalues
    5. Sorting the eigenvectors by decreasing eigenvalues
    6. Choosing k eigenvectors with the largest eigenvalues
    7. Transforming the samples onto the new subspace

    """
    num_data = inputs.shape[0]
    num_dims = inputs.shape[1]
    # 1. Calculate means for each class
    data_of_class = {}
    for i in range(1, 6):
        indecies = (labels == i)
        data_of_class[i] = inputs[indecies]
    mean_of_class = {}
    for key in data_of_class.keys():
        mean_of_class[key] = np.mean(data_of_class[key], axis=0)
    total_mean = np.mean(inputs, axis=0)
    # 2. Compute the Covariance Matrix between classes
    S_b = np.cov([inputs[:, i] for i in range(num_dims)])
    # 3. Compute the Covariance Matrix within classes
    S_w = np.zeros(S_b.shape)
    for key in data_of_class.keys():
        tmp = data_of_class[key] - mean_of_class[key]
        S_w += np.dot(tmp.T, tmp)
    # 4. Computing eigenvectors and corresponding eigenvalues
    mat = np.dot(np.linalg.pinv(S_w), S_b)
    eig_vals, eig_vecs = np.linalg.eig(mat)
    # 5. Sorting the eigenvectors by decreasing eigenvalues
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(num_dims)]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # 6. Choosing k eigenvectors with the largest eigenvalues
    matrix_w = np.stack([eig_pairs[i][1]
                         for i in range(num_components)], axis=1)
    # 7. Transforming the samples onto the new subspace
    return np.dot(inputs, matrix_w)


def lda_plot(inputs, labels, with_legend=True, num_components=2, file_name='default_lda'):
    # LDA
    lda_data = lda_fit_n_transform(
        inputs, labels, num_components=num_components)
    # Visualization
    fig, ax = plt.subplots()
    scatter_obj = []
    for i in range(5):
        indecies = (labels == (i+1))
        scatter_obj.append(ax.scatter(
            lda_data[indecies, 0], lda_data[indecies, 1], c=colors[i], s=8.0, alpha=0.5))
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
    lda_plot(data['X_train'], data['T_train'])


if __name__ == '__main__':
    main()