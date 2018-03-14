""" regularized Least Square Error
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Matrix import Mat


class rLSE(object):
    """ regularized Least Square Error
    """

    def __init__(self, file_path, num_bases, lambda_):
        """ Get polynomail regression result by least square error

        Args
        ----
        file_path : str,
            path to dataset
        num_bases : int,
            degree of polynomial
        lambda_ : float,
            parameters of L2 regularizar term 
        """
        # data IO
        assert num_bases >= 2
        self._num_bases = int(num_bases)
        self._lambda = float(lambda_)
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
        self._error = self._mse()
        # print('input shape = {}'.format(self._input.shape))
        # print('label shape = {}'.format(self._label.shape))

    def _fit(self):
        """ W = (A.t() * A + lambda * I).inv() * A.t() * b
            A : input
            b : label
        """
        I_mat = []
        for i in range(self._num_bases):
            temp = []
            for j in range(self._num_bases):
                v = 1.0 if i == j else 0.0
                temp.append(v)
            I_mat.append(temp)
        I_mat = Mat(I_mat)
        return (self._input.t() * self._input + self._lambda * I_mat).inv() * self._input.t() * self._label

    def _mse(self):
        """ mean square error

        Retrun
        ------
        mean square error : float, shape=()
        """
        error = self._input * self._weights - self._label
        sum_ = 0.0
        for i in range(self._input.shape[0]):
            sum_ += error[i, 0]**2
        return sum_/self._input.shape[0]

    def show_formula(self):
        """ to print out result of LSE as formula:
            y = Wn * x^n + W(n-1) * x^(n-1) + ... + W1 * x^1 + W0
        """
        formula = "LSE Result : y = "
        for base in range(self._num_bases-1, -1, -1):
            if base == 0:
                formula += "{}".format(self._weights[base, 0])
            else:
                formula += "{} * x^{} + ".format(self._weights[base, 0], base)
        print(formula)

    def show_error(self):
        """ to print out error of LSE
        """
        print('LSE Error : {}'.format(self._error))


def main():
    rlse = rLSE('data.txt', num_bases=2, lambda_=0.3)
    rlse.show_formula()
    rlse.show_error()


if __name__ == '__main__':
    main()
