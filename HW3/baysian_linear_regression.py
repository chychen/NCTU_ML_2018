""" 
3. Baysian Linear regression
      INPUT: the precision (i.e., b) for initial prior w ~ N(0, b-1I) and all other required inputs for the polynomial basis linear model geneartor (1.b)
      FUNCTION: call 1.b to generate one data point, and update the prior. and calculate the paramters of predictive distribution, repeat until the posterior probability converges.
      OUTPUT: print the new data point and the current paramters for posterior and predictive distribution.
      HINT: Online learning
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
from Generator import Generator
from Matrix import Mat


def get_data_var(y):
    # mean
    sum_ = 0.0
    for v in range(y.shape[0]):
        sum_ += y[v, 0]
    mean = sum_/y.shape[0]
    # var
    sum_ = 0.0
    for v in range(y.shape[0]):
        sum_ += (y[v, 0]-mean)**2
    var = sum_/y.shape[0]
    return var


def baysian_linear_regression(b, num_basis, error_variance):
    """
    Prior
        b: precision of prior weights (b*I is linear independent matrix), shape=(num_basis, num_basis)
        S: inverse of prior corvariance matrix (S is a covariance matrix), shape=(num_basis, num_basis)
        m: mean of prior, shape=(num_basis, 1)
    Posterior
        X: design matrix, shape=(n, num_basis)
        y: all data samples, shape=(n, 1)
        a: precision of data distribution, shape=(num_basis, num_basis)
        COV = a*X.t()*X + b*I, shape=(num_basis, num_basis)
        mean = COV.inv()*(a*X.t()*y+Sm), shape=(num_basis, 1)
    Predictive Distribution
        mean = X*mu
        variance = 1/a+X*COV.inv()*X.t()
    """
    prior_weight_var = b * Mat.identity(dims=num_basis)
    weights = []
    for i in range(num_basis):
        weights.append(Generator.univariate_gaussian(
            mean=0, variance=prior_weight_var[i, i]))

    prior_mean_mat = Mat([weights]).t()
    prior_cov_mat = b.inv()
    posterior_mean_mat = math.inf
    posterior_cov_mat = math.inf
    diff = math.inf
    X = None
    y = None
    iteration_idx = 0
    while diff > 1e-2:
        iteration_idx += 1
        new_y, new_x = Generator.polynomial(num_basis, error_variance, weights)
        if iteration_idx == 1:
            y = Mat([[new_y]])
            X = Mat([[new_x**i for i in range(num_basis)]])
        else:
            y.append([new_y])
            X.append([new_x**i for i in range(num_basis)])
        # change annotation
        a = error_variance
        S = prior_cov_mat.inv()
        m = prior_mean_mat
        # posterior
        posterior_cov_mat = a*X.t()*X + S
        COV = posterior_cov_mat
        posterior_mean_mat = COV.inv()*(a*X.t()*y+S*m)
        # check converge
        if iteration_idx>1:
            sub_mat = prior_mean_mat - posterior_mean_mat
            diff = 0.0
            for i in range(sub_mat.shape[0]):
                diff += abs(sub_mat[i, 0])
            diff /= sub_mat.shape[0]
        # predictive distribution
        predictive_mean = X*posterior_mean_mat
        predictive_var= 1/a+X*posterior_cov_mat.inv()*X.t()
        # update prior
        prior_mean_mat = posterior_mean_mat
        prior_cov_mat = posterior_cov_mat

        print('Iteration: {}'.format(iteration_idx))
        print('New Data Point: x:{}, y:{}'.format(new_x, new_y))
        print('Posterior Mean:\n{}'.format(posterior_mean_mat))
        print('Posterior Variance:\n{}'.format(posterior_cov_mat))
        print('Predictive Distribution Mean:\n{}'.format(predictive_mean))
        print('Predictive Distribution Variance:\n{}'.format(predictive_var))
        if iteration_idx % 5 == 0:
            input()


def main():
    num_basis = 2
    b_matrix = Mat.identity(num_basis)
    baysian_linear_regression(b_matrix, num_basis, 3)


if __name__ == '__main__':
    main()
