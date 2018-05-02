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
import numpy as np
from utils import load_mnist
from utils import ConfusionMatrix
from Tensor import Tensor
from Matrix import Mat


def mean_abs_diff(a, b):
    """ calculate the mean of absolute difference between a and b
    """
    diff = 0.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            diff += abs(a[i, j]-b[i, j])
    return diff/(a.shape[0]*a.shape[1])


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


def em_algorithm(train_images, theta, lambda_, iterations=1e6, epsilon=1e-15):
    """
    Expectation step:
        theta : shape=[10, image_size]
            the probability of class i showing 1 for each pixels
        lambda = [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9]
        M : number of pixel=1 in each pixels
        image_size : total number of pixels = 28*28
        N : number of data
        mu_pixel : mu of each pixel showing 1 for each class
        P(Zi=z,Xi|theta) = lambda(z) * theta[z]**M[i] * (1-theta[z])**(N-M)
        all(z) = sum_over_i(P(Zi=z,Xi|theta))
        W(i,z) = P(Zi=z,Xi|theta)/all(z)
    Maximization step:
        lambda(z) = sum_over_i(W(i,z))/N
        theta(z) = sum_over_i(W(i,z)*M[i])/sum_over_i(W(i,z))
    """
    N = train_images.shape[0]
    image_size = train_images.shape[1] * train_images.shape[2]
    # M = []
    # for i in range(N):
    #     tmp = 0
    #     for row_idx in range(train_images.shape[1]):
    #         for col_idx in range(train_images.shape[2]):
    #             tmp += train_images[i, row_idx, col_idx]
    #     M.append(tmp)

    for iter_idx in range(int(3)):
        print('iteration index: ', iter_idx)
        old_theta = copy.deepcopy(theta)
        theta = Mat.zeros(shape=[10, image_size]) + 1e-8 # avoid zeros
        sum_w = [1e-8 for _ in range(10)]
        # Expectation Step
        for i in range(N):
            weights = Mat.zeros([train_images.shape[0], 10])
            for z in range(10):
                logprob = 0.0
                for row_idx in range(train_images.shape[1]):
                    for col_idx in range(train_images.shape[2]):
                        logprob += math.log(old_theta[z, row_idx*28+col_idx])*train_images[i, row_idx, col_idx] + math.log(
                            1-old_theta[z, row_idx*28+col_idx])*(1-train_images[i, row_idx, col_idx])
                        
                    logprob += math.log(lambda_[z])
                    weights[i, z] = logprob
                    # input(weights[i, z])
            max_weight = [-1e10 for _ in range(N)]

            for z in range(10):
                if max_weight[i] < weights[i,z]:
                    max_weight[i] = weights[i,z]

            sum_ = 0.0
            for z in range(10):
                weights[i,z] -= max_weight[i] # normalized
                weights[i,z] = math.exp(weights[i,z])
                # input(weights[i,z])
                sum_ += weights[i,z]
            for z in range(10):
                weights[i,z] /= sum_
                
            for z in range(10):
                sum_w[z] += weights[i, z]
                for pixels in range(image_size):
                    theta[z, pixels] += weights[i, z] * \
                        train_images[i, pixels//28, pixels % 28]

        # Maximization Step
        for z in range(10):
            lambda_[z] = sum_w[z]/N
            for pixels in range(image_size):
                theta[z, pixels] /= sum_w[z]
                if theta[z, pixels]>=1.0:
                    # input(theta[z, pixels])
                    theta[z, pixels] =1.0-1e-8
        print(lambda_)
        print(theta)
        # if mean_abs_diff(old_theta, theta) < epsilon:
        #     break

    return weights

    # cluster_n = 10
    # px_n = 784
    # img_n = 100
    # alpha = 0.00000008

    # pi = [0.1 for i in range(cluster_n)]
    # mu = [[np.random.uniform(0, 1) for i in range(px_n)] for j in range(cluster_n)]
    # z  = [[0.1 for i in range(cluster_n)] for j in range(img_n)]

    # for idx in range(20):

    #     Nm = [0 for i in range(cluster_n)]
    #     mean = [[0.0 for i in range(px_n)] for j in range(cluster_n)]
    #     predict = [-1 for i in range(img_n)]
        
    #     for n in range(img_n):

    #         # Expectation step
    #         for k in range(cluster_n):
    #             for i in range(px_n):
    #                 z[n][k] += train_images[n, i//28, i%28] * math.log(mu[k][i]) + (1 - train_images[n, i//28, i%28]) * math.log(1.0 - mu[k][i])
    #             z[n][k] += math.log(pi[k])

    #         z_0 = z[n][0]
    #         # divided by first element
    #         temp_max = max(z[n])
    #         temp_max_idx = 0
    #         for k in range(cluster_n):
    #             if temp_max == z[n][k]:
    #                 temp_max_idx = k

    #         for k in range(cluster_n):
    #             z[n][k] -= z[n][temp_max_idx]
                
    #         for k in range(cluster_n):
    #             try:
    #                 z[n][k] = math.exp(z[n][k])
    #             except:
    #                 z[n][k] = 0.0
            
    #         # normalize
    #         z_sum = sum(z[n])
    #         z[n] = [z_k / z_sum for z_k in z[n]]

    #         # predict
    #         temp_max = 0
    #         predict_k = 0
    #         for k in range(1, cluster_n):
    #             if z[n][k] > temp_max:
    #                 temp_max = z[n][k]
    #                 predict_k = k
                    
    #         predict[n] = predict_k

    #         # Maximization step
    #         for k in range(cluster_n):
    #             Nm[k] += z[n][k]
    #             for i in range(px_n):
    #                 mean[k][i] += z[n][k] * train_images[n, i//28, i%28]
        
    #     for k in range(cluster_n):
    #         pi[k] = float(Nm[k] + alpha) / float(img_n + alpha * cluster_n)
    #         for i in range(px_n):
    #             mean[k][i] += alpha
    #             mean[k][i] /= Nm[k] + alpha * img_n
    #     mu = mean
    #     print(mean)
    # return Mat(z)


def main():
    print('Start data I/O...')
    train_images, train_labels = load_mnist(
        dataset='training', fetch_size=100)
    # test_images, test_labels = load_mnist(dataset='testing', fetch_size=10000)

    train_images = preprocessing(train_images)
    init_theta = Mat([[np.random.uniform(0.0, 1.0) for _ in range(train_images.shape[1]
                                          * train_images.shape[2])] for _ in range(10)])
    # init_theta = Mat([[0.5 for _ in range(train_images.shape[1]
    #                                       * train_images.shape[2])] for _ in range(10)])
    init_lambda = [0.1 for _ in range(10)]
    logits = em_algorithm(train_images, init_theta, init_lambda)
    prediction = []
    for i in range(logits.shape[0]):
        largest_idx = 0
        for z in range(logits.shape[1]):
            print(logits[i, z])
            if logits[i, z] > logits[i, largest_idx]:
                largest_idx = z
        # input()
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
