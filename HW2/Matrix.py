from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


class Mat():
    """ implementation for matrix operation
    dtype = Mat(Customized)
    """

    def __init__(self, matrix, pad=5):
        if isinstance(matrix, list):
            self._rows = len(matrix)
            self._cols = len(matrix[0])
            self._mat = copy.deepcopy(matrix)
        elif isinstance(matrix, Mat):
            self._rows = matrix.shape[0]
            self._cols = matrix.shape[1]
            self._mat = copy.deepcopy(matrix._mat)
        self._pad = pad

    def __getitem__(self, tuple_index):
        if isinstance(tuple_index, int):
            return self._mat[tuple_index]
        elif len(tuple_index) == 2:
            row, col = tuple_index
            return self._mat[row][col]

    def __setitem__(self, tuple_index, value):
        row, col = tuple_index
        self._mat[row][col] = value

    def __add__(self, rhs):
        """ override the function of addition

        Args
        ----
        rhs : Mat(Customized)
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Mat(self._mat)
            for _r in range(self._rows):
                for _c in range(self._cols):
                    result[_r, _c] += float(rhs)
            return result

        # type check
        if not isinstance(rhs, Mat):
            raise ValueError('only accept dtype: Mat(Customized)')
        result = Mat(self._mat)
        # shape check
        assert result.shape == rhs.shape
        for _r in range(self._rows):
            for _c in range(self._cols):
                result[_r, _c] += rhs[_r, _c]
        return result

    def __sub__(self, rhs):
        """ override the function of subtraction

        Args
        ----
        rhs : Mat(Customized)
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Mat(self._mat)
            for _r in range(self._rows):
                for _c in range(self._cols):
                    result[_r, _c] -= float(rhs)
            return result

        # type check
        if not isinstance(rhs, Mat):
            raise ValueError('only accept dtype: Mat(Customized)')
        result = Mat(self._mat)
        # shape check
        assert result.shape == rhs.shape
        for _r in range(self._rows):
            for _c in range(self._cols):
                result[_r, _c] -= rhs[_r, _c]
        return result

    def __mul__(self, rhs):
        """ override the function of multiplication

        Args
        ----
        rhs : Mat(Customized)
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Mat(self._mat)
            for _r in range(self._rows):
                for _c in range(self._cols):
                    result[_r, _c] *= float(rhs)
            return result
        else:
            raise ValueError('only accept dtype: scalar')

    def __repr__(self):
        repr_ = "[\n"
        for _r0 in range(self._rows):
            temp_1 = "["
            for _r1 in range(self._cols):
                temp_1 += (str(self._mat[_r0][_r1]) + ", ").rjust(self._pad)
            repr_ += temp_1 + "], \n"
        repr_ += "]\n"
        return repr_

    @property
    def shape(self):
        """ represent shape as (rows, cols)
        """
        return (self._rows, self._cols)

    @property
    def dtype(self):
        """ represent data dype
        """
        return 'Mat(Customized)'

    # override
    __rmul__ = __mul__
    __radd__ = __add__
