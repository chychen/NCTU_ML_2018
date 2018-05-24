from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
import numpy as np
import matplotlib.pyplot as plt


def length(vec, axis):
    return np.sqrt(np.sum(vec**2, axis=axis))


def kmeans(data, num_clusters, epsilon=1e-2):
    # range
    max_x = np.amax(data[:, 0])
    min_x = np.amin(data[:, 0])
    max_y = np.amax(data[:, 1])
    min_y = np.amin(data[:, 1])
    # random init data classes
    cluster_center = np.array([max_x-min_x, max_y-min_y]) * \
        np.random.random((num_clusters, 2))+np.array([min_x, min_y])
    num_cluster_table = np.zeros(shape=[num_clusters])
    r_one_hot = np.zeros(shape=[data.shape[0], num_clusters])
    r_index = np.zeros(shape=[data.shape[0]], dtype=np.int) + \
        (num_clusters*np.random.random(data.shape[0])).astype(np.int)
    # training
    iterator_counter = 0
    # show
    plt.clf()
    plt.ion()
    plt.scatter(data[:, 0], data[:, 1], c=r_index, alpha=0.5)
    plt.show(block=False)
    plt.pause(0.01)
    while True:
        iterator_counter += 1
        print(iterator_counter)
        old_center = copy.deepcopy(cluster_center)
        # E step
        for k in range(num_clusters):
            num_cluster_table[k] = np.count_nonzero(r_index == k)
            if num_cluster_table[k] == 0:
                # pseudo count
                num_cluster_table[k] = 1
        for n in range(data.shape[0]):
            min_ = np.inf
            best_k = -1
            for k in range(num_clusters):
                dist = length(data[n]-cluster_center[k], axis=0)
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
        # show
        plt.clf()
        plt.ion()
        plt.scatter(data[:, 0], data[:, 1], c=r_index, alpha=0.5)
        plt.show(block=False)
        plt.pause(0.01)
        # terminal condition
        if all(length(cluster_center - old_center, axis=1) < epsilon):
            print('STOP')
            plt.show(block=True)
            break


def main():
    data = []
    with open("data/circle.txt") as f:
        for line in f.readlines():
            data.append([float(line.strip().split(',')[0]),
                         float(line.strip().split(',')[1])])
    data = np.array(data)
    print('data.shape', data.shape)
    kmeans(data, num_clusters=2)

    data = []
    with open("data/moon.txt") as f:
        for line in f.readlines():
            data.append([float(line.strip().split(',')[0]),
                         float(line.strip().split(',')[1])])
    data = np.array(data)
    print('data.shape', data.shape)
    kmeans(data, num_clusters=2)


if __name__ == "__main__":
    main()
