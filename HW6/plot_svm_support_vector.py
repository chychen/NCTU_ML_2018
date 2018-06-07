""" 
1. need to export the libsvm path to the env variables
export PYTHONPATH="${PYTHONPATH}:/Users/Jay/Desktop/NCTU_ML_2018/HW6/libsvm/python"
export PYTHONPATH="${PYTHONPATH}:/Users/Jay/Desktop/NCTU_ML_2018/HW6/libsvm/tools"
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from libsvm.python.svmutil import *
from libsvm.tools.grid import *
from svmutil import *
from grid import *
from pca import pca_plot_with_svm
from data_loader import load_mnist
import numpy as np

def test_kernel(mode):
    """
    mode==0 : linear
    mode==1 : polynomial
    mode==2 : rbf
    """
    # svm
    y, x = svm_read_problem('data/training.csv')
    m = svm_train(y, x, '-s 0 -t {} -c 1'.format(mode))
    # get support vector
    support_vectors = m.get_SV()
    p_labels, p_accs, p_vals = svm_predict(np.zeros(len(support_vectors)), support_vectors, m)
    # sparse to dense
    dense_sv = np.zeros(shape=[len(support_vectors), 28*28])
    for i, dict_ in enumerate(support_vectors):
        for key in dict_.keys():
            dense_sv[i, key] = dict_[key]
    # vis
    data = load_mnist()
    pca_plot_with_svm(data['X_train'], data['T_train'], dense_sv, np.array(p_labels), file_name='svm_pca_mode{}'.format(mode))

def precomputed_kernel():
    data = load_mnist()

    def RBF_kernel(x1, x2, gamma):
        return np.exp(-1.0*gamma*np.sum((x1-x2)**2, axis=0))

    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    def new_kernel(x1, x2, gamma):
        return linear_kernel(x1, x2) + RBF_kernel(np.array(x1), np.array(x2), gamma)

    # sparse
    gram_matrix = []
    for i, x1 in enumerate(data['X_train']):
        print(i)
        tmp = {}
        tmp[0] = i+1
        for j, x2 in enumerate(data['X_train']):
            tmp[j+1] = new_kernel(x1, x2, gamma=1.0/28)
        gram_matrix.append(tmp)
    test_gram_matrix = []
    for i, x1 in enumerate(data['X_test']):
        print(i)
        tmp = {}
        tmp[0] = i  # any number
        for j, x2 in enumerate(data['X_train']):
            tmp[j+1] = new_kernel(x1, x2, gamma=1.0/28)
        test_gram_matrix.append(tmp)

    # train
    prob = svm_problem(data['T_train'], gram_matrix)
    param = svm_parameter('-s 0 -t 4 -c 1')
    m = svm_train(prob, param)
    # get support vector
    support_vectors = m.get_SV()
    p_labels, p_accs, p_vals = svm_predict(np.zeros(len(support_vectors)), support_vectors, m)
    # sparse to dense
    dense_sv = np.zeros(shape=[len(support_vectors), 28*28])
    for i, dict_ in enumerate(support_vectors):
        for key in dict_.keys():
            dense_sv[i, key] = dict_[key]
    # vis
    pca_plot_with_svm(data['X_train'], data['T_train'], dense_sv, np.array(p_labels), file_name='svm_pca_mode_precomputed')


def main():
    # linear, polynomial, RBF
    # for mode in range(3):
    #     test_kernel(mode=mode)
    precomputed_kernel()


if __name__ == "__main__":
    main()
