from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from Matrix import Mat


class Tensor():
    """ implementation for tensor operation, rank 3
    dtype = Tensor(Customized)
    """

    def __init__(self, tensor, pad=5):
        if isinstance(tensor, list):
            self._tensor = copy.deepcopy(tensor)
            assert isinstance(
                tensor[0], Mat), "items in tensor must be type Mat(Customized)"
            self._r0, self._r1, self._r2 = len(
                tensor), tensor[0].shape[0], tensor[0].shape[1]
        elif isinstance(tensor, Tensor):
            self._tensor = copy.deepcopy(tensor._tensor)
            self._r0, self._r1, self._r2 = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        self._pad = pad

    def __getitem__(self, tuple_index):
        if isinstance(tuple_index, int):
            return self._tensor[tuple_index]
        elif len(tuple_index) == 2:
            r, c = tuple_index
            return self._tensor[r][c]
        elif len(tuple_index) == 3:
            r0, r1, r2 = tuple_index
            return self._tensor[r0][r1, r2]

    def __setitem__(self, tuple_index, value):
        r0, r1, r2 = tuple_index
        self._tensor[r0][r1, r2] = value

    def __add__(self, rhs):
        """ override the function of addition

        Args
        ----
        rhs : Mat(Customized)
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Tensor(self._tensor)
            for _r0 in range(self._r0):
                result[_r0] += float(rhs)
            return result

        # type check
        if not isinstance(rhs, Tensor):
            raise ValueError('only accept dtype: Tensor(Customized)')
        result = Tensor(self._tensor)
        # shape check
        assert result.shape == rhs.shape
        for _r0 in range(self._r0):
            result[_r0] += rhs[_r0]
        return result

    def __sub__(self, rhs):
        """ override the function of subtraction

        Args
        ----
        rhs : Mat(Customized)
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Tensor(self._tensor)
            for _r0 in range(self._r0):
                result[_r0] -= float(rhs)
            return result

        # type check
        if not isinstance(rhs, Tensor):
            raise ValueError('only accept dtype: Tensor(Customized)')
        result = Tensor(self._tensor)
        # shape check
        assert result.shape == rhs.shape
        for _r0 in range(self._r0):
            result[_r0] -= rhs[_r0]
        return result

    def __mul__(self, rhs):
        """ override the function of multiplication

        Args
        ----
        rhs : Mat(Customized)
        """
        # if scalar
        if isinstance(rhs, float) or isinstance(rhs, int):
            result = Tensor(self._tensor)
            for _r0 in range(self._r0):
                result[_r0] *= float(rhs)
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
                    temp_2 += (str(self._tensor[_r0]
                                   [_r1, _r2]) + ", ").rjust(self._pad)
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
    input_ = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    temp = Mat(input_)
    tensor = []
    tensor.append(temp)
    tensor.append(temp*10)
    tensor.append(temp*111.222)
    ten = Tensor(tensor)
    print(ten)
    print(ten.shape)


if __name__ == '__main__':
    main()
