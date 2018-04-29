from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import struct
import array

from Tensor import Tensor
from Matrix import Mat


class ConfusionMatrix(object):
    """ calculate the sensitivity and specificity to evaluate the classification model.
    """

    def __init__(self, logits, labels):
        """
        Inputs
        ------
        logits : Mat(Customized), shape=(n,1)
        labels : Mat(Customized), shape=(n,1)
        """
        assert isinstance(logits, Mat)
        assert isinstance(labels, Mat)

        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.N = logits.shape[0]
        for i in range(logits.shape[0]):
            if int(logits[i, 0]) == 0:
                if int(labels[i, 0]) == 0:
                    self.TN += 1
                elif int(labels[i, 0]) == 1:
                    self.FN += 1
            elif int(logits[i, 0]) == 1:
                if int(labels[i, 0]) == 0:
                    self.FP += 1
                elif int(labels[i, 0]) == 1:
                    self.TP += 1

    def show_matrix(self):
        print('Confusion Matrix')
        print('N : {}'.format(self.N))
        print('True Positive : {}\tFalse Negative : {}'.format(self.TP, self.FN))
        print('False Positive : {}\tTrue Negative : {}'.format(self.FP, self.TN))

    def show_sensitivity(self):
        print('Sensitivity/Recall/HitRate')
        print(self.TP/(self.TP+self.FN))

    def show_specificity(self):
        print('Specificity')
        print(self.TN/(self.FP+self.TN))

    def show_accuracy(self):
        print('Accuracy')
        print((self.TP+self.TN)/self.N)


def load_mnist(dataset, fetch_size, path="."):
    """ load mnist and transform bytes into numpy array

    Args
    ----
    dataset : str,
        'training' or 'testing' or raise ValueError
    path : str, default='.'
        path to dataset

    Returns
    -------
    images : Tensor(Customized), shape=(?, 28, 28)
    labels : uint8, shape=(?,)

    # ref: https://docs.python.org/3/library/struct.html#format-characters
    # ref: https://docs.python.org/3/library/array.html#module-array
    # 'I' unsigned int
    # '>' big-endian
    # 'B' unsigned char
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(fname_img, 'rb') as fimg:
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        assert fetch_size <= size
        img = array.array("B", fimg.read())

    with open(fname_lbl, 'rb') as flbl:
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        assert fetch_size <= size
        lbl = array.array("B", flbl.read())

    image_list = []
    labels = []
    for i in range(fetch_size):
        image_temp = []
        temp = img[i*rows*cols:(i+1)*rows*cols]
        for j in range(rows):
            image_temp.append(temp[j*rows:j*rows+cols])
        labels.append(lbl[i])
        image_list.append(image_temp)
    images = Tensor(image_list)
    return images, labels


def main():
    train_images, train_labels = load_mnist(
        dataset='training', fetch_size=60000)
    test_images, test_labels = load_mnist(dataset='testing', fetch_size=10000)
    print(test_images[0])
    print(test_labels[0])


if __name__ == '__main__':
    main()
