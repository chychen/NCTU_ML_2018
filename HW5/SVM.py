""" 
1. need to export the libsvm path to the env variables
export PYTHONPATH="${PYTHONPATH}:/Users/Jay/Desktop/NCTU_ML_2018/HW5/libsvm/python"
export PYTHONPATH="${PATH}:/Users/Jay/Desktop/NCTU_ML_2018/HW5/libsvm/tools"
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from libsvm.python.svmutil import *
from libsvm.tools.grid import *
from svmutil import *
from grid import *
import numpy as np


def test_kernel():
    y, x = svm_read_problem('data/training.csv')
    ty, tx = svm_read_problem('data/testing.csv')
    m = svm_train(y, x, '-s 0 -t 2 -c 1')
    p_label, p_acc, p_val = svm_predict(y, x, m)
    p_label, p_acc, p_val = svm_predict(ty, tx, m)


def grid_search():
    rate, param = find_parameters(
        'data/training.csv', '-log2c -5,15,2 -log2g 3,-15,-2 -v 5')
    print('rate', rate)
    print('param', param)


def precomputed_kernel():
    data = {}
    # input
    for file_name in ['X_train', 'X_test']:
        with open('data/'+file_name+'.csv') as f:
            tmp_data = []
            for line in f.readlines():
                tmp = list(map(lambda x: float(x), line.strip().split(',')))
                tmp_data.append(tmp)
            data[file_name] = tmp_data[:10]
    # label
    for file_name in ['T_train', 'T_test']:
        with open('data/'+file_name+'.csv') as f:
            tmp_data = []
            for line in f.readlines():
                tmp = int(line)
                tmp_data.append(tmp)
            data[file_name] = tmp_data[:10]

    def RBF_kernel(x1, x2, gamma):
        return np.exp(-1.0*gamma*np.mean((x1-x2)**2, axis=0))

    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    def new_kernel(x1, x2, gamma):
        return RBF_kernel(np.array(x1), np.array(x2), gamma) + np.dot(np.array(x1), np.array(x2))
    
    # after grid serach we found c=3.0 gamma=-5.0 can get highest 98.5%
    gram_matrix = []
    for i, x1 in enumerate(data['X_train']):
        print(i)
        tmp = [i]
        for x2 in data['X_train']:
            tmp.append(new_kernel(x1, x2, gamma=-5.0))
        gram_matrix.append(tmp)

    # train
    prob = svm_problem(data['T_train'], gram_matrix, isKernel=True)
    param = svm_parameter('-s 0 -t 4 -c 3')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(data['T_test'], data['X_test'], model)


def main():
    # test_kernel()
    # grid_search()
    precomputed_kernel()


if __name__ == "__main__":
    main()
