from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
import numpy as np
import matplotlib.pyplot as plt


def length(vec, axis):
    return np.sqrt(np.sum(vec**2, axis=axis))


def kmeans(data, kernel_fn, num_clusters, epsilon=1e-1):
    # range
    max_x = np.amax(data[:, 0])
    min_x = np.amin(data[:, 0])
    max_y = np.amax(data[:, 1])
    min_y = np.amin(data[:, 1])
    # init center
    cluster_center = np.array([max_x-min_x, max_y-min_y]) * \
        np.random.random((num_clusters, 2))+np.array([min_x, min_y])
    r_one_hot = np.zeros(shape=[data.shape[0], num_clusters])
    r_index = np.zeros(shape=[data.shape[0]], dtype=np.int) + \
        (num_clusters*np.random.random(data.shape[0])).astype(np.int)
    num_cluster_table = np.zeros(shape=[num_clusters])
    r_one_hot[np.arange(data.shape[0], dtype=np.int), r_index] = 1.0
    GramMatrix = np.zeros(shape=[data.shape[0], data.shape[0]])
    for p in range(data.shape[0]):
        for q in range(data.shape[0]):
            GramMatrix[p, q] = kernel_fn(data[p], data[q])
    # training
    iterator_counter = 0
    while True:
        iterator_counter += 1
        print(iterator_counter)
        # show
        plt.clf()
        plt.ion()
        plt.scatter(data[:, 0], data[:, 1], c=r_index, alpha=0.5)
        plt.show(block=False)
        plt.pause(0.01)
        old_center = copy.deepcopy(cluster_center)
        # E step
        # || fi(X_n) - MU_k ||
        #  = kernel_fn(X_n, X_n) - 2/num_clusters*sum_over_j(r_jk*kernel_fn(X_n, X_j)) + 1/(num_clusters**2)*sum_over_p_q(GramMatrix_p_q)
        # GramMatrix_p_q = r_np*r_nq*K(X_p,X_q)
        for k in range(num_clusters):
            num_cluster_table[k] = np.count_nonzero(r_index == k)
            if num_cluster_table[k] == 0:
                # pseudo count
                num_cluster_table[k] = 1
        sum_over_p_q = np.zeros(shape=[num_clusters])
        for k in range(num_clusters):
            for p in range(data.shape[0]):
                for q in range(data.shape[0]):
                    sum_over_p_q[k] += r_one_hot[p, k] * \
                        r_one_hot[q, k]*GramMatrix[p, q]
        for n in range(data.shape[0]):
            min_ = np.inf
            best_k = -1
            for k in range(num_clusters):
                sum_over_j = 0.0
                for j in range(data.shape[0]):
                    sum_over_j += r_one_hot[j, k]*GramMatrix[n, j]
                dist = GramMatrix[n, n] - 2.0/num_cluster_table[k] * \
                    sum_over_j + 1.0/(num_cluster_table[k]**2)*sum_over_p_q[k]
                if dist < min_:
                    min_ = dist
                    best_k = k
            r_index[n] = best_k
        r_one_hot[np.arange(data.shape[0], dtype=np.int), r_index] = 1.0
        # M step
        for k in range(num_clusters):
            cluster_center[k] = np.sum(
                data * r_one_hot[:, k:k+1], axis=0) / num_cluster_table[k]
        print(cluster_center)

        if all(length(cluster_center - old_center, axis=1) < epsilon):
            break


def RBF_kernel(x1, x2, sigma=0.5):
    return np.exp(-1.0*np.sum((x1-x2)**2, axis=0)/(2*sigma**2))

def sigmoid_kernel(x1, x2, gamma=1.0, r=1.0):
    return np.tanh(gamma*np.dot(x1, x2) +r)

def no_kernel(x1, x2):
    return np.dot(x1, x2)


def main():
    data = []
    # with open("data/circle.txt") as f:
    with open("data/moon.txt") as f:
        for line in f.readlines():
            data.append([float(line.strip().split(',')[0]),
                         float(line.strip().split(',')[1])])
    data = np.array(data)
    print('data.shape', data.shape)
    # kmeans(data, kernel_fn=no_kernel, num_clusters=2)
    kmeans(data, kernel_fn=RBF_kernel, num_clusters=2)
    # kmeans(data, kernel_fn=sigmoid_kernel, num_clusters=2)


if __name__ == "__main__":
    main()
