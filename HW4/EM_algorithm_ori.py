"""
2. EM algorithm 
 
INPUT: 
MNIST training data and label sets. 

 FUNCTION: 
 a. Binning the gray level value into two bins. Treating all pixels as random variables following Bernoulli distributions. 
 Note that each pixel follows a different Binomial distribution independent to others. 

 b. Use EM algorithm to cluster each image into ten groups. You should come up with a reasonable rule to determine convergence. 
 (a simple run out of the loop should be used as the ultimatum) 

 OUTPUT: 
For each digit, output a confusion matrix and the sensitivity and specificity of the clustering applied to the training data.
"""
import math
import copy
from utils import load_mnist
from utils import ConfusionMatrix
from Tensor import Tensor
from Matrix import Mat


def mean_abs_diff(a, b):
    """ calculate the mean of absolute difference between a and b
    """
    diff = 0.0
    for v_a, v_b in zip(a, b):
        diff += abs(v_a-v_b)
    return diff/len(a)


def preprocessing(train_images):
    """ binning the gray level value into two bins as binomial distribution
    """
    for data_idx in range(train_images.shape[0]):
        for row_idx in range(train_images.shape[1]):
            for col_idx in range(train_images.shape[2]):
                if train_images[data_idx, row_idx, col_idx] < 128.0:
                    train_images[data_idx, row_idx, col_idx] = 0
                else:
                    train_images[data_idx, row_idx, col_idx] = 1
    return train_images


def em_algorithm(train_images, theta, lambda_, iterations=1e6, epsilon=1e-12):
    """
    Expectation step:
        lambda = [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9]
        theta = [P0,P1,P2,P3,P4,P5,P6,P7,P8,P9]
            Pi : the probability of class i showing 1
        M : number of pixel=1 in each image
        image_size : total number of pixels = 28*28
        N : number of data
        P(Zi=z,Xi|theta) = lambda(z) * theta[z]**M[i] * (1-theta[z])**(N-M)
        all(z) = sum_over_i(P(Zi=z,Xi|theta))
        W(i,z) = P(Zi=z,Xi|theta)/all(z)
    Maximization step:
        lambda(z) = sum_over_i(W(i,z))/N
        theta(z) = sum_over_i(W(i,z)*M[i])/sum_over_i(W(i,z))
    """
    # tally the value=1 in each image
    image_size = train_images.shape[1] * train_images.shape[2]
    N = train_images.shape[0]
    # M = [100 for _ in range(N)]
    M = []
    for i in range(N):
        tmp = 0
        for row_idx in range(train_images.shape[1]):
            for col_idx in range(train_images.shape[2]):
                tmp += train_images[i, row_idx, col_idx]
        M.append(tmp)
    for iter_idx in range(int(iterations)):
        print('iteration index: ', iter_idx)
        old_theta = copy.deepcopy(theta)
        # Expectation Step
        weights = Mat.zeros([N, 10])
        for i in range(N):
            sum_ = 0.0
            for z in range(10):
                # prob = lambda_[z] * theta[z]**M[i] * (1-theta[z])**(N-M[i])
                if theta[z] ==0:
                    logprob = 1.0
                elif theta[z]==1:
                    logprob = 1.0
                else:
                    logprob = math.log(
                        lambda_[z]) + math.log(theta[z])*M[i] + math.log(1-theta[z])*(image_size-M[i])
                prob = math.exp(logprob)
                sum_ += prob
                weights[i, z] = prob
            for z in range(10):
                weights[i, z] /= sum_
            #     print(weights[i, z])
            # input()

        # # find most possible prediction
        # predicts = []
        # for i in range(N):
        #     largest = 0.0
        #     predict = 0
        #     for z in range(10):
        #         if weights[i, z]>largest:
        #             largest = weights[i,z]
        #             predict = z
        #         predicts.append(predict)

        # Maximization Step
        for z in range(10):
            sum_w = 1e-10
            sum_w_x = 1e-10
            for i in range(N):
                sum_w += weights[i, z]
                sum_w_x += weights[i, z]*M[i]/image_size
            lambda_[z] = sum_w/N
            theta[z] = sum_w_x/sum_w
        print(lambda_)
        print(theta)
        if mean_abs_diff(old_theta, theta) < epsilon:
            break
        # update M


    return weights


def main():
    print('Start data I/O...')
    train_images, train_labels = load_mnist(
        dataset='training', fetch_size=600)
    # test_images, test_labels = load_mnist(dataset='testing', fetch_size=10000)

    train_images = preprocessing(train_images)
    import numpy as np
    # init_theta = [np.random.uniform(0.2, 0.8) for _ in range(10)]
    init_theta = [0.5 for _ in range(10)]
    init_lambda = [0.1 for _ in range(10)]
    logits = em_algorithm(train_images, init_theta, init_lambda)
    prediction = []
    for i in range(logits.shape[0]):
        largest_idx = 0
        for z in range(logits.shape[1]):
            if logits[i, z] > logits[i, largest_idx]:
                largest_idx = z
        prediction.append([largest_idx])
    prediction = Mat(prediction)
    train_labels = Mat([train_labels]).t()
    # confusion matrix
    for z in range(10):
        print('digit {}'.format(z))
        tmp_logits = Mat(prediction)
        tmp_labels = Mat(train_labels)
        for i in range(tmp_logits.shape[0]):
            if tmp_logits[i, 0] == tmp_labels[i, 0]:
                tmp_logits[i, 0] = 1
            else:
                tmp_logits[i, 0] = 0
            if tmp_labels[i, 0] == z:
                tmp_labels[i, 0] = 1
            else:
                tmp_labels[i, 0] = 0
        CM = ConfusionMatrix(tmp_logits, tmp_labels)
        CM.show_matrix()
        CM.show_accuracy()
        CM.show_sensitivity()
        CM.show_specificity()


if __name__ == '__main__':
    main()
