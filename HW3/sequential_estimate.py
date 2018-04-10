""" 
2. Sequential estimate the mean and variance from the data given from the univariate gaussian data generator (1.a).
        NOTE: you should derive the recursive function of mean and variance based on the sequential esitmation. 
        INPUT: m, s as in (1.a)
        FUNCTION: call (1.a) to get a new data point from N(m, s), use sequential estimation to find the current estimates to m and s., repeat until the estimates converge.
        OUTPUT: print the new data point and the current estimiates of m and s in each iteration.
        HINT: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

Sample Mean:
Xbar(n) = Xbar(n-1) + (X(n)-Xbar(n-1))/n
Sample Variance:
S(n) = (n-2)/(n-1)*S(n-1) + (X(n)-Xbar(n-1))**2/n 

Avoid numerical instability
M(2,n) = sum(X-Xbar)**2
M(2,n) = M(2,n-1) + (X(n)-Xbar(n-1))*(X(n)-Xbar(n))
S(n) = M(2,n)/n-1
var(n) - M(2,n)/n
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
from Generator import Generator


def sequential_estimate(mean, variance):
    estimated_mean = math.inf
    estimated_variance = math.inf
    M = math.inf
    iteration_idx = 0
    while abs(estimated_mean - mean)>1e-2: #  or abs(estimated_variance-variance)>1e-1
        iteration_idx += 1
        data_point = Generator.univariate_gaussian(mean, variance)
        if iteration_idx==1:
            estimated_mean = data_point
            estimated_variance = 0.0
            M = 0.0
        else:
            last_estimated_mean = estimated_mean
            estimated_mean = estimated_mean + (data_point-estimated_mean)/iteration_idx
            M = M + (data_point-last_estimated_mean)*(data_point-estimated_mean)
            estimated_variance = M/(iteration_idx-1)
        print('Iteration: {}'.format(iteration_idx))
        print('New Data Point: {}'.format(data_point))
        print('Estimated Mean: {}'.format(estimated_mean))
        print('Estimated Variance: {}'.format(estimated_variance))
        if iteration_idx%1000==0:
            input()

def main():
    sequential_estimate(0, 3)

if __name__ == '__main__':
    main()