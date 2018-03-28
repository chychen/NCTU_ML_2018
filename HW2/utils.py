from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import struct
import array

from Tensor import Tensor


def load_mnist(dataset, path="."):
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
        img = array.array("B", fimg.read())

    with open(fname_lbl, 'rb') as flbl:
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = array.array("B", flbl.read())

    image_list = []
    labels = []
    for i in range(size):
        image_temp = []
        temp = img[i*rows*cols:(i+1)*rows*cols]
        for j in range(rows):
            image_temp.append(temp[j*rows:j*rows+cols])
        labels.append(lbl[i])
        image_list.append(image_temp)
    images = Tensor(image_list)
    return images, labels


def main():
    train_images, train_labels = load_mnist(dataset='training')
    test_images, test_labels = load_mnist(dataset='testing')
    print(test_images[0])
    print(test_labels[0])


if __name__ == '__main__':
    main()
