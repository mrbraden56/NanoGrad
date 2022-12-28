from slim_grad.engine.tensor import Tensor
import random

#Matrix([tensor(5), tensor(3)])
class Matrix:
    def __init__(self, data, shape=None) -> None:
        if isinstance(data, list): self.data = self.tensorfy(data)
        else: self.data=data
        self.shape=shape

    def __repr__(self) -> str:
        return f"Matrix(data={self.data})"

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx]=val

    def tensorfy(self, data):
        def get_shape(lst):
            if not isinstance(lst, list):
                return []
            shape = [len(lst)]
            subshape = get_shape(lst[0])
            return shape + subshape
        shape=get_shape(data)
        for i in range(shape[0]):
            for j in range(shape[1]):
                data[i][j]=Tensor(data[i][j])
        return data

    @classmethod
    def zeros(cls, shape):
        weights = [ [Tensor(0)] * shape[1] for i in range(shape[0]) ]
        return cls(weights, shape)

    @classmethod
    def rand_uniform(cls, shape):
        weights=cls.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                weights[i][j]=random.uniform(-1, 1)
        return cls(weights, shape)

    #TODO: Make sure dot product accumlates gradients
    @classmethod
    def dot(cls, x, y):
        if x.shape()[1]!=y.shape()[0]: raise Exception("Incorrect array size for dot product")
        product=cls.zeroes(cls, [x.shape()[0], y.shape()[1]])
        y_column=[]
        for row_idx, x_row in enumerate(x.data):
            for col_idx, j in enumerate(range(y.shape()[1])):#col
                for i in range(y.shape()[0]):#row
                    y_column.append(y[i][j])
                product[row_idx][col_idx]=sum(r*c for r,c in zip(x_row, y_column))
                y_column.clear()
        return product

    @staticmethod
    def add_bias(x, y):
        for i in range(x.shape()[0]):
            x[i]=[xi+yi for xi,yi in zip(x[i], y.data[0])]
        return x 