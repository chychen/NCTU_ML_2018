from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def load_mnist():
    data = {}
    # input
    for file_name in ['X_train', 'X_test']:
        with open('data/'+file_name+'.csv') as f:
            tmp_data = []
            for line in f.readlines():
                tmp = list(map(lambda x: float(x), line.strip().split(',')))
                tmp_data.append(tmp)
            data[file_name] = np.array(tmp_data)
    # label
    for file_name in ['T_train', 'T_test']:
        with open('data/'+file_name+'.csv') as f:
            tmp_data = []
            for line in f.readlines():
                tmp = int(line)
                tmp_data.append(tmp)
            data[file_name] = np.array(tmp_data)
    return data


def main():
    pass


if __name__ == '__main__':
    main()
