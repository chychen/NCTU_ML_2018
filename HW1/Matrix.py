from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


class Mat():
    """ implementation for matrix operation
    dtype = Mat(Customized)
    """

    def __init__(self, matrix, pad=20):
        if isinstance(matrix, list):
            self._rows = len(matrix)
            self._cols = len(matrix[0])
            self._mat = copy.deepcopy(matrix)
        elif isinstance(matrix, Mat):
            self._rows = matrix.shape[0]
            self._cols = matrix.shape[1]
            self._mat = copy.deepcopy(matrix._mat)
        self._pad = pad

    def LU_decompose(self):
        assert self._rows == self._cols, "only square matrix could be invrsed"
        # init
        mat = []
        for _ in range(self._rows):
            temp = []
            for _ in range(self._cols):
                temp.append(0.0)
            mat.append(temp)
        L_mat = Mat(mat)
        U_mat = Mat(mat)
        for i in range(self._rows):
            L_mat[i, i] = 1.0  # Identity
        # decompose
        for i in range(self._rows):
            for j in range(self._cols):
                if j >= i:
                    # Upper Matrix
                    temp = self._mat[i][j]
                    for k in range(0, i):
                        temp -= L_mat[i, k] * U_mat[k, j]
                    U_mat[i, j] = temp
                else:
                    # Lower Matrix
                    temp = self._mat[i][j]
                    for k in range(0, j):
                        temp -= L_mat[i, k] * U_mat[k, j]
                    L_mat[i, j] = temp / U_mat[j, j]
        return L_mat, U_mat

    def inv(self):
        """ inverse matrix by LU decomposition 
        A = L * U
        U^-1 * L^-1 = A^-1
        """
        # decompose matrix
        L_mat, U_mat = self.LU_decompose()
        # inverse L
        L_mat_inv = Mat(L_mat)
        for i in range(self._rows):
            for j in range(self._cols):
                if i > j:
                    temp = 0.0
                    for k in range(i):
                        temp -= L_mat_inv[i, k] * L_mat_inv[k, j]
                    L_mat_inv[i, j] = temp
        # inverse U
        I_mat = []
        for i in range(self._rows):
            temp = []
            for j in range(self._cols):
                v = 1.0 if i == j else 0.0
                temp.append(v)
            I_mat.append(temp)

        U_mat_inv = Mat(I_mat)
        for _r in range(self._rows):
            # make sure pivot == 0
            pivot = U_mat[_r, _r]
            for _c in range(_r, self._cols):
                U_mat_inv[_r, _c] /= pivot
                U_mat[_r, _c] /= pivot
        for _r in range(self._rows):
            for _c in range(self._cols):
                if _c > _r and U_mat[_r, _c] != 0.0:
                    factor = U_mat[_r, _c]
                    U_mat_inv[_r, _c] -= factor * U_mat_inv[_c, _c]
                    for j in range(_c, self._cols):
                        U_mat[_r, j] -= factor * U_mat[_c, j]

        return U_mat_inv*L_mat_inv

    def t(self):
        """ return transpose matrix
        """
        t_mat = []
        for _r in range(self._cols):
            temp = []
            for _c in range(self._rows):
                temp.append(self._mat[_c][_r])
            t_mat.append(temp)
        return Mat(t_mat)

    def __getitem__(self, tuple_index):
        row, col = tuple_index
        return self._mat[row][col]

    def __setitem__(self, tuple_index, value):
        row, col = tuple_index
        self._mat[row][col] = value

    def __add__(self, matrix):
        """ override the function of addition

        Args
        ----
        matrix : Mat(Customized)
        """
        # if scalar
        if isinstance(matrix, float) or isinstance(matrix, int):
            result = Mat(self._mat)
            for _r in range(self._rows):
                for _c in range(self._cols):
                    result[_r, _c] += float(matrix)
            return result

        # type check
        if not isinstance(matrix, Mat):
            raise ValueError('only accept dtype: Mat(Customized)')
        result = Mat(self._mat)
        # shape check
        assert result.shape == matrix.shape
        for _r in range(self._rows):
            for _c in range(self._cols):
                result[_r, _c] += matrix[_r, _c]
        return result

    def __sub__(self, matrix):
        """ override the function of addition

        Args
        ----
        matrix : Mat(Customized)
        """
        # if scalar
        if isinstance(matrix, float) or isinstance(matrix, int):
            result = Mat(self._mat)
            for _r in range(self._rows):
                for _c in range(self._cols):
                    result[_r, _c] -= float(matrix)
            return result

        # type check
        if not isinstance(matrix, Mat):
            raise ValueError('only accept dtype: Mat(Customized)')
        result = Mat(self._mat)
        # shape check
        assert result.shape == matrix.shape
        for _r in range(self._rows):
            for _c in range(self._cols):
                result[_r, _c] -= matrix[_r, _c]
        return result
            

    def __mul__(self, matrix):
        """ override the function of multiplication

        Args
        ----
        matrix : Mat(Customized)
        """
        # if scalar
        if isinstance(matrix, float) or isinstance(matrix, int):
            result = Mat(self._mat)
            for _r in range(self._rows):
                for _c in range(self._cols):
                    result[_r, _c] *= float(matrix)
            return result

        # type check
        if not isinstance(matrix, Mat):
            raise ValueError('only accept dtype: Mat(Customized)')
        # shape check
        assert self._cols == matrix.shape[0]
        # init
        result = []
        for _ in range(self._rows):
            temp = []
            for _ in range(matrix.shape[1]):
                temp.append(0.0)
            result.append(temp)
        # mul
        for _r in range(self._rows):
            for _c in range(matrix.shape[1]):
                for left_v, right_v in zip(self._mat[_r], matrix.t()[_c, :]):
                    result[_r][_c] += (left_v * right_v)
        return Mat(result)

    def __repr__(self):
        pad = self._pad
        repr_ = "".rjust(pad)
        for i in range(self._cols):
            repr_ += "<c{}>".format(i).rjust(pad)
        repr_ += "\n"

        for i in range(self._rows):
            temp = "<r{}>".format(i).rjust(pad)
            for j in range(self._cols):
                temp += (str(self._mat[i][j])).rjust(pad)
            repr_ += temp + "\n"
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
