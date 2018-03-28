"""
1. Naive Bayes classifier 
  Create a Naive Bayes classifier for each handwritten digit that support discrete and continuous features 
  INPUT: 
           i.Training image data from MNIST (http://yann.lecun.com/exdb/mnist/) 
  Pleaes read the description in the link to understand the format. 
  Basically, each image is represented by 28X28X8bits (the header is in big endian format; you need to deal with it), you can use a char arrary to store an image. 
  There are some headers you need to deal with as well, please read the link for more details. 
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import math
from utils import load_mnist
from Tensor import Tensor


def show_error_rate(result_log_posteriors, test_data, test_labels):
    """ find the max one as prediction result, statistic the error rate according to labels

    Args
    ----
    result_log_posteriors : list of list, shape=[num_data, 10]
    """
    error_counter = 0
    predictions = [-1 for _ in range(test_data.shape[0])]
    for idx in range(test_data.shape[0]):
        max_temp = -math.inf
        for label in range(10):
            if result_log_posteriors[idx][label] > max_temp:
                max_temp = result_log_posteriors[idx][label]
                predictions[idx] = label
        if predictions[idx] != test_labels[idx]:
            error_counter += 1
    print('Error Rate: {}%'.format(error_counter/test_data.shape[0]*100.0))


def discrete_naive_bayes(train_data, train_labels, test_data, test_labels):
    """ Tally the frequency of the values of each pixel into 32 bins. Perform Naive Bayes classifer.
    Note that to avoid empty bin, you can use a peudocount (such as the minimum value in other bins) for instead.

    Args
    ----
    train_data : Tensor(Customized), shape=[num_data,28,28]
    train_labels : list, shape=[num_data]
    test_data : Tensor(Customized), shape=[num_data,28,28]
    test_labels : list, shape=[num_data]

    Output
    ------
    Print out the the posterior (in log scale to avoid underflow) of the ten categories (0-9) for each row in INPUT 3 (your prediction is the category having the highest posterior),
    and tally the number of correct prediction by comparing with INPUT4. Calculate and report the error rate in the end.
    """
    # training part
    print('Tally the frequency from Training Data...')
    # 1. tally the frequency of each pixel of each bins by transfering training data pixel value into discrete bins, and statistic the prior of each category
    bins_table = [Tensor.zeros(shape=[28, 28, 32]) +
                  1e-4 for _ in range(10)]  # avoid zero
    labels_table = [0 for _ in range(10)]
    for idx in range(train_data.shape[0]):
        if idx % 1000 == 0:
            print('Progress... {}/{}'.format(idx, train_data.shape[0]))
        label = train_labels[idx]
        labels_table[label] += 1
        for row in range(train_data.shape[1]):
            for col in range(train_data.shape[2]):
                bin_idx = train_data[idx, row, col]//32
                bins_table[label][row, col, bin_idx] += 1
    for label in range(10):
        bins_table[label] /= labels_table[label]
    # testing part
    print('Inference the posteriros on Testing Data...')
    # 1. inference the posterior for all category, and print out
    result_log_posteriors = [
        [0.0 for _ in range(10)] for _ in range(test_data.shape[0])]
    for idx in range(test_data.shape[0]):
        if idx % 1000 == 0:
            print('Progress... {}/{}'.format(idx, test_data.shape[0]))
        for label in range(10):
            for row in range(test_data.shape[1]):
                for col in range(test_data.shape[2]):
                    bin_idx = test_data[idx, row, col]//32
                    result_log_posteriors[idx][label] += math.log(
                        bins_table[label][row, col, bin_idx])
            # mul with prior (1.0/labels_table[label])
            result_log_posteriors[idx][label] += math.log(
                1.0/labels_table[label])
            # print(result_log_posteriors[idx][label])
    # 2. find the max one as prediction result, statistic the error rate according to labels
    show_error_rate(result_log_posteriors, test_data, test_labels)


def continuous_naive_bayes(train_data, train_labels, test_data, test_labels):
    """ Use MLE to fit a Gaussian distribution for the value of each pixel. Perform Naive Bayes classifer.

    Output
    ------
    Print out the the posterior (in log scale to avoid underflow) of the ten categories (0-9) for each row in INPUT 3 (your prediction is the category having the highest posterior),
    and tally the number of correct prediction by comparing with INPUT4. Calculate and report the error rate in the end.
    """
    # training part
    print('Tally the frequency from Training Data...')
    # 1. tally the frequency of each pixel by gaussian distribution, and statistic the prior of each category
    sum_table = [Tensor.zeros(shape=[28, 28, 1]) for _ in range(10)]
    labels_table = [0 for _ in range(10)]
    for idx in range(train_data.shape[0]):
        if idx % 1000 == 0:
            print('Progress... {}/{}'.format(idx, train_data.shape[0]))
        label = train_labels[idx]
        labels_table[label] += 1
        for row in range(train_data.shape[1]):
            for col in range(train_data.shape[2]):
                sum_table[label][row, col, 0] += train_data[idx, row, col]
    print('Tally the mean and variance...')
    mean_table = [Tensor.zeros(shape=[28, 28, 1]) for _ in range(10)]
    for label in range(10):
        # mean, shape=[10][28, 28, 1]
        mean_table[label] = sum_table[label] / labels_table[label]
    # variance, shape=[10][28, 28, 1]
    variance_table = [Tensor.zeros(shape=[28, 28, 1])+1e-4 for _ in range(10)]
    for idx in range(train_data.shape[0]):
        if idx % 1000 == 0:
            print('Progress... {}/{}'.format(idx, train_data.shape[0]))
        label = train_labels[idx]
        for row in range(train_data.shape[1]):
            for col in range(train_data.shape[2]):
                variance_table[label][row, col,
                                      0] += (train_data[idx, row, col] - mean_table[label][row, col, 0])**2
    for label in range(10):
        variance_table[label] = (variance_table[label] / labels_table[label])
    # testing part
    print('Inference the posteriros on Testing Data...')
    # 1. inference the posterior for all category, and print out
    result_log_posteriors = [
        [0.0 for _ in range(10)] for _ in range(test_data.shape[0])]
    for idx in range(test_data.shape[0]):
        if idx % 1000 == 0:
            print('Progress... {}/{}'.format(idx, test_data.shape[0]))
        for label in range(10):
            for row in range(test_data.shape[1]):
                for col in range(test_data.shape[2]):
                    pixel_value = test_data[idx, row, col]
                    # gaussian_p = 1.0/math.sqrt(2*math.pi*variance_table[label][row, col, 0])*math.exp(-(
                    #     pixel_value-mean_table[label][row, col, 0])**2/(2*variance_table[label][row, col, 0]))
                    # # log prob # avoid log(zero)
                    # result_log_posteriors[idx][label] += math.log(
                    #     gaussian_p+1e-8)
                    result_log_posteriors[idx][label] -= (pixel_value-mean_table[label][row, col, 0])**2/(
                        2*variance_table[label][row, col, 0])*1.0/(math.sqrt(2*math.pi*variance_table[label][row, col, 0]))
            # mul with prior (1.0/labels_table[label])
            # log prob
            result_log_posteriors[idx][label] += math.log(
                1.0/labels_table[label])
    # 2. find the max one as prediction result, and statistic the error rate according to labels
    show_error_rate(result_log_posteriors, test_data, test_labels)


def main():
    print('Start data I/O...')
    train_images, train_labels = load_mnist(
        dataset='training', fetch_size=60000)
    test_images, test_labels = load_mnist(dataset='testing', fetch_size=10000)

    print('Start analysis...')
    if ARGS.mode == 'discrete':
        discrete_naive_bayes(train_images, train_labels,
                             test_images, test_labels)
    elif ARGS.mode == 'continuous':
        continuous_naive_bayes(train_images, train_labels,
                               test_images, test_labels)
    else:
        raise ValueError('{} is not a valid mode'.format(ARGS.mode))
    return


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("mode", type=str,
                        help="\'discrete\' or \'continuous\'")
    ARGS = PARSER.parse_args()

    main()
