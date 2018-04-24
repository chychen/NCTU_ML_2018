"""
1. Logistic regression 

INPUT: 
n (number of data point, D), 
mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 (m: mean, v: variance) 

FUNCTION: 

 a. Generate n data point: D1= {(x1, y1), (x2 ,y2), ..., (xn, yn) }, 
where x and y are independently sampled from N(mx1, vx1) and N(my1, vy1) respectively. 
(use the Gaussian random number generator you did for homework 3.). 

 b. Generate n data point: D2= {(x1, y1), (x2 ,y2), ..., (xn, yn) }, 
where x and y are independently sampled from N(mx2, vx2) and N(my2, vy2) respectively. 

 c. Use Logistic regression to separate D1 and D2. 
You should implement both Newton's and steepest gradient descent method during optimization, 
i.e., when the Hessian is singular, use steepest descent for instead. 
You should come up with a reasonable rule to determine convergence. 
(a simple run out of the loop should be used as the ultimatum) 

 OUTPUT: 
The confusion matrix and the sensitivity and specificity of the logistic regression applied to the training data D. 
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import math
from Generator import Generator
from Matrix import Mat
from utils import ConfusionMatrix


def is_33_singullar(mat):
    det = mat[0, 0]*mat[1, 1]*mat[2, 2]+mat[1, 0]*mat[2, 1]*mat[0, 2]+mat[0, 1]*mat[1, 2]*mat[2,
                                                                                              0] - mat[0, 2]*mat[1, 1]*mat[2, 0]+mat[0, 1]*mat[1, 0]*mat[2, 2]+mat[0, 0]*mat[2, 1]*mat[1, 2]
    print('Determinant: ', det)
    return det == 0


def mean_abs_diff(a, b):
    """ calculate the mean of absolute difference between a and b
    """
    return (a-b).mean()


def steepest_gradient_descent(weights, inputs, labels, iterations=1e6, epsilon=1e-4, learning_rate=1e-3):
    """ steepest gradient descent
    to classify data into two classes (Bernoulli Distribution)
    (gradient in matrix form)
    A : inputs, shape=(n,k=3)
    yi : labels[i,0], shape=()
    xi : inputs[i:i+1], shape=(1,k=3)
    W : weights, shape=(k=3,1)
    gradients : A.t()*(yi-1/(1+e**(-xi*W)), shape=(k,1)
    W(n+1) = W(n) + learning_rate*gradients
    """
    for idx in range(int(iterations)):
        print('iteration index: ', idx)
        old_weights = Mat(weights)
        gradients = []
        for i in range(labels.shape[0]):
            exp_term = (-1*inputs[i:i+1]*old_weights)[0, 0]
            gradients.append([labels[i, 0]-1.0/(1+math.e**exp_term)])
        gradients = learning_rate*inputs.t()*Mat(gradients)
        weights = old_weights + gradients
        print(weights)
        if mean_abs_diff(old_weights, weights) < epsilon:
            break

    return weights


def newton_method(weights, inputs, labels, iterations=1e6, epsilon=1e-4, learning_rate=1e-3):
    """ newton's method
    to classify data into two classes (Bernoulli Distribution)
    (gradient in matrix form)
    A : inputs, shape=(n,k=3)
    yi : labels[i,0], shape=()
    xi : inputs[i:i+1], shape=(1,k=3)
    W : weights, shape=(k=3,1)
    gradients : A.t()*(yi-1/(1+e**(-xi*W)), shape=(k,1)
    D : diagonal matrix, shape=(n,n)
        trace(i)=e**(-xi*W)/(1+e**(-xi*W))**2
    hessian : A.t()*D*A
    W(n+1) = W(n) + gradients
    """
    for idx in range(int(iterations)):
        print('iteration index: ', idx)
        old_weights = Mat(weights)
        gradients = []
        for i in range(labels.shape[0]):
            exp_term = (-1*inputs[i:i+1]*old_weights)[0, 0]
            gradients.append([labels[i, 0]-1.0/(1+math.e**exp_term)])
        D = Mat.identity(dims=labels.shape[0])
        for i in range(labels.shape[0]):
            exp_term = (-1*inputs[i:i+1]*old_weights)[0, 0]
            D[i, i] = math.e**exp_term/((1+math.e**exp_term)**2)
        gradients = inputs.t()*Mat(gradients)
        hessian = inputs.t()*D*inputs
        if is_33_singullar(hessian):
            print('Singular')
            break
        grad = learning_rate*hessian.inv()*gradients
        for i in range(grad.shape[0]):
            if grad[i, 0] > 0 and grad[i, 0] > 1:
                grad[i, 0] = 1
            if grad[i, 0] < 0 and grad[i, 0] < -1:
                grad[i, 0] = -1
        weights = old_weights - grad
        print(weights)
        if mean_abs_diff(old_weights, weights) < epsilon:
            break
    return weights


def inference(weights, inputs):
    logits = []
    for i in range(inputs.shape[0]):
        exp_term = (-1*inputs[i:i+1]*weights)[0, 0]
        output = 1.0/(1+math.e**exp_term)
        print(output)
        # decision boundary
        if output > 0.5:
            output = 1
        else:
            output = 0
        logits.append([output])
    return Mat(logits)


def logistic_regression(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, optimizer='SGD'):
    """
    weights : shape=(k=3,1)
    Input
    -----
    optimizer : 'SGD' or 'NTM'
        'SGD' == 'Steepest Gradient Descent'
        'NTM' == 'Newton's Method'
    """
    inputs = []
    labels = []
    D1_label = 0.0
    D2_label = 1.0
    bias_term = 1.0
    for _ in range(n):
        # Data 1
        D1x = Generator.univariate_gaussian(mx1, vx1)
        D1y = Generator.univariate_gaussian(my1, vy1)
        inputs.append([bias_term, D1x, D1y])
        labels.append([D1_label])
        # Data 2
        D2x = Generator.univariate_gaussian(mx2, vx2)
        D2y = Generator.univariate_gaussian(my2, vy2)
        inputs.append([bias_term, D2x, D2y])
        labels.append([D2_label])
    inputs = Mat(inputs)
    labels = Mat(labels)
    # init weights
    weights = Mat([[1.0], [1.0], [1.0]])
    print('inputs shape:\t', inputs.shape)
    print('labels shape:\t', labels.shape)
    print('weights shape:\t', weights.shape)
    # optimization
    if optimizer == 'SGD':
        weights = steepest_gradient_descent(weights, inputs, labels)
    elif optimizer == 'NTM':
        weights = newton_method(weights, inputs, labels)
    else:
        raise AttributeError('{} is not a valid optimizor'.format(optimizer))
    # inference
    logits = inference(weights, inputs)
    # evaluate model
    CM = ConfusionMatrix(logits, labels)
    CM.show_matrix()
    CM.show_accuracy()
    CM.show_sensitivity()
    CM.show_specificity()


def main():
    logistic_regression(10, 3, 1, 3, 1, 10, 1, 4, 1, optimizer='SGD')
    logistic_regression(10, 3, 1, 3, 1, 10, 1, 4, 1, optimizer='NTM')


if __name__ == '__main__':
    main()
