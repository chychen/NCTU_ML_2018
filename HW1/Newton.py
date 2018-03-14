""" Newton's Method Optimization
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Matrix import Mat


class NewtonMethod(object):
    """ Newton's Method Optimization
    """

    def __init__(self, file_path, num_bases, converge_epsilon, init_weights):
        """ Get polynomail regression result by Newton's Method Optimization

        Args
        ----
        file_path : str,
            path to dataset
        num_bases : int,
            degree of polynomial
        converge_epsilon : float,
            the condition of convergence
        init_weights : list, shape=[num_bases,]
            the parameters of polynomial
        """
        assert num_bases == len(init_weights)
        self._num_bases = int(num_bases)
        self._converge_epsilon = float(converge_epsilon)
        self._init_weights = Mat([[float(i)] for i in init_weights])
        self._data = []
        self._input = []
        self._label = []
        with open(file_path, 'r') as file_:
            for line in file_.readlines():
                # 1 for x^0 (bias)
                self._data.append([1, int(line.strip().split(',')[0])])
                self._label.append([int(line.strip().split(',')[1])])
        for v in self._data:
            for i in range(2, num_bases):
                v += [v[1]**i]
            self._input.append(v)
        self._input = Mat(self._input)
        self._label = Mat(self._label)
        self._weights = self._fit()
        # print('input shape = {}'.format(self._input.shape))
        # print('label shape = {}'.format(self._label.shape))

    def _fit(self):
        """ 
        d_F = 2 * (A.t() * A * X - A.t() * b)
        dd_F = 2 * A.t() * A
        """
        loss = 1e10
        weights = self._init_weights
        while loss > self._converge_epsilon:
            d_F = 2 * (self._input.t() * self._input *
                       weights - self._input.t() * self._label)
            dd_F = 2 * self._input.t() * self._input
            weights = weights - dd_F.inv() * d_F
            loss = self._mse(weights)
            print('Error : {}'.format(loss))
        return weights

    def _mse(self, weights):
        """ mean square error

        Retrun
        ------
        mean square error : float, shape=()
        """
        error = self._input * weights - self._label
        sum_ = 0.0
        for i in range(self._input.shape[0]):
            sum_ += error[i, 0]**2
        return sum_ / self._input.shape[0]

    def show_formula(self):
        """ to print out result of NewtonMethod as formula:
            y = Wn * x^n + W(n-1) * x^(n-1) + ... + W1 * x^1 + W0
        """
        formula = "y = "
        for base in range(self._num_bases-1, -1, -1):
            if base == 0:
                formula += "{}".format(self._weights[base, 0])
            else:
                formula += "{} * x^{} + ".format(self._weights[base, 0], base)
        print(formula)


def main():
    ntm = NewtonMethod('data.txt', num_bases=4,
                       converge_epsilon=0.1, init_weights=[1 for _ in range(4)])
    ntm.show_formula()


if __name__ == '__main__':
    main()
