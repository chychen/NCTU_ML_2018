from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


class Tensor():
    """ implementation for tensor operation, rank 3
    dtype = Tensor(Customized)
    """

    def __init__(self, tensor, pad=5):
        if isinstance(tensor, list):
            self._tensor = copy.deepcopy(tensor)
            self._r0, self._r1, self._r2 = len(
                tensor), len(tensor[0]), len(tensor[0][0])
        elif isinstance(tensor, Tensor):
            self._tensor = copy.deepcopy(tensor._tensor)
            self._r0, self._r1, self._r2 = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        self._pad = pad

    @classmethod
    def zeros(cls, shape):
        """ init zeros tensor with shape
        """
        assert len(shape) == 3, "shape must have 3 ranks"
        r0, r1, r2 = shape
        temp = [[[0 for _ in range(r2)] for _ in range(r1)] for _ in range(r0)]
        return Tensor(temp)

    def __getitem__(self, tuple_index):
        if isinstance(tuple_index, int):
            return self._tensor[tuple_index][:][:]
        elif len(tuple_index) == 2:
            r, c = tuple_index
            return self._tensor[r][c][:]
        elif len(tuple_index) == 3:
            r0, r1, r2 = tuple_index
            return self._tensor[r0][r1][r2]

    def __setitem__(self, tuple_index, value):
        r0, r1, r2 = tuple_index
        self._tensor[r0][r1][r2] = value

    def __add__(self, rhs):
        """ override the function of addition

        Args
        ----
        rhs : scalar or Tensor
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Tensor(self._tensor)
            for _r0 in range(self._r0):
                for _r1 in range(self._r1):
                    for _r2 in range(self._r2):
                        result[_r0, _r1, _r2] += float(rhs)
            return result

        # type check
        if not isinstance(rhs, Tensor):
            raise ValueError('only accept dtype: Tensor(Customized)')
        result = Tensor(self._tensor)
        # shape check
        assert result.shape == rhs.shape
        for _r0 in range(self._r0):
            for _r1 in range(self._r1):
                for _r2 in range(self._r2):
                    result[_r0, _r1, _r2] += rhs[_r0, _r1, _r2]
        return result

    def __sub__(self, rhs):
        """ override the function of subtraction

        Args
        ----
        rhs : scalar or Tensor
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Tensor(self._tensor)
            for _r0 in range(self._r0):
                for _r1 in range(self._r1):
                    for _r2 in range(self._r2):
                        result[_r0, _r1, _r2] -= float(rhs)
            return result

        # type check
        if not isinstance(rhs, Tensor):
            raise ValueError('only accept dtype: Tensor(Customized)')
        result = Tensor(self._tensor)
        # shape check
        assert result.shape == rhs.shape
        for _r0 in range(self._r0):
            for _r1 in range(self._r1):
                for _r2 in range(self._r2):
                    result[_r0, _r1, _r2] -= rhs[_r0, _r1, _r2]
        return result

    def __truediv__(self, rhs):
        """ override the function of true div

        Args
        ----
        rhs : scalar or Tensor
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Tensor(self._tensor)
            for _r0 in range(self._r0):
                for _r1 in range(self._r1):
                    for _r2 in range(self._r2):
                        result[_r0, _r1, _r2] /= float(rhs)
            return result
        else:
            raise ValueError('only accept dtype: scalar')

    def __mul__(self, rhs):
        """ override the function of multiplication

        Args
        ----
        rhs : scalar or Tensor
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Tensor(self._tensor)
            for _r0 in range(self._r0):
                for _r1 in range(self._r1):
                    for _r2 in range(self._r2):
                        result[_r0, _r1, _r2] *= float(rhs)
            return result
        else:
            raise ValueError('only accept dtype: scalar')

    def __repr__(self):
        repr_ = "[\n"
        for _r0 in range(self._r0):
            temp_1 = "[\n"
            for _r1 in range(self._r1):
                temp_2 = "[ "
                for _r2 in range(self._r2):
                    temp_2 += (str(self._tensor[_r0][_r1]
                                   [_r2]) + ", ").rjust(self._pad)
                temp_1 += temp_2 + "], \n"
            repr_ += temp_1 + "], \n"
        repr_ += "]\n"
        return repr_

    @property
    def shape(self):
        """ represent shape as (rows, cols)
        """
        return (self._r0, self._r1, self._r2)

    @property
    def dtype(self):
        """ represent data dype
        """
        return 'Tensor(Customized)'

    # override
    __rmul__ = __mul__
    __radd__ = __add__


def main():
    tensor = [[[i*j*k for i in range(3)] for j in range(3)] for k in range(4)]
    ten = Tensor(tensor)
    print(ten)
    print(ten.shape)
    print(Tensor.zeros([2, 3, 4]))
    print(Tensor.zeros([2, 3, 4]).shape)
    print((Tensor.zeros([2, 3, 4]) + 1) / 3)


if __name__ == '__main__':
    main()
