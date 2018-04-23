""" Random Date Generator 
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np


class Generator(object):
    """ Random Date Generator 
    """

    @classmethod
    def univariate_gaussian(cls, mean, variance):
        """ univariate gaussian data generator 
        INPUT: expectation value (m), variance (s)
        OUTPUT: one outcome ~ N(m, s)
        HINT: https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
        NOTE: 1. you have to handcraft your geneartor based on one of the approaches given in the hint.
                    2. you can use uniform distribution function

        generating by Box–Muller method
        X = sqrt(-2*log(U))*cos(2*pi*V)
        U, V are uniform distribution
        """
        U = np.random.uniform(0.0, 1.0)
        V = np.random.uniform(0.0, 1.0)
        ret = math.sqrt(-2*math.log2(U))*math.cos(2*math.pi*V)
        # denormalized
        ret = ret * math.sqrt(variance) + mean
        return ret

    @classmethod
    def polynomial(cls, number_basis, variance, weight):
        """ polynomial basis linear model (y = WTPhi(x)+e ; e ~ N(0, a)) data generator
        INPUT: basis number (n; ex. n=2 -> y = w0x0 + w1x1), a, w
        OUTPUT: y
        NOTE: there is an internal constraint: -10.0 < x < 10.0, x is uniformly distributed.
        """
        X = np.random.uniform(-10.0, 10.0)
        assert isinstance(weight, list)
        assert len(weight) == number_basis
        y = 0.0
        for v in range(number_basis):
            y += (weight[v] * X ** v)
        y += cls.univariate_gaussian(0, variance)
        return y, X


if __name__ == '__main__':
    print(Generator.univariate_gaussian(3, 1))
    print(Generator.polynomial(3, 1, [3, 4, 1]))
