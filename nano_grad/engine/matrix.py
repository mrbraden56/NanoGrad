import os
from nano_grad.engine.tensor import Tensor

class Matrix:
    def __init__(self, data, shape=None, backend="python") -> None:
        self.data=data
        self.shape=shape
        self.backend=backend

    def __repr__(self) -> str:
        return f"Matrix(data={self.data})"

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx]=val

    @classmethod
    def array(cls, data):
        def get_shape(lst):
            if not isinstance(lst, list):
                return []
            shape = [len(lst)]
            subshape = get_shape(lst[0])
            return shape + subshape
        shape=get_shape(data)
        if len(shape)==1:
            for i in range(shape[0]):
                data[i]=Tensor(data[i])
        elif len(shape)==2:
            for i in range(shape[0]):
                if len(shape)==1: 
                    data[i]=Tensor(data[i])
                    continue
                for j in range(shape[1]):
                    data[i][j]=Tensor(data[i][j])
        return cls(data, shape)

    @classmethod
    def zeros(cls, shape):
        if len(shape)==1:
            weights=[Tensor(0)] * shape[0]
            if type(shape)!=list: shape=[shape]
        elif len(shape)==2: 
            weights = [ [Tensor(0)] * shape[0] for i in range(shape[1]) ]
        return cls(weights, shape)

    @classmethod
    def dot(cls, x, y):
        if x.shape[1]!=y.shape[0]: raise Exception("Incorrect array size for dot product")
        product=cls.zeros([x.shape[0], y.shape[1]])
        y_column=[]
        for row_idx, x_row in enumerate(x.data):
            for col_idx, j in enumerate(range(y.shape[1])):#col
                for i in range(y.shape[0]):#row
                    y_column.append(y[i][j])
                product[row_idx][col_idx]=cls.sum([r*c for r,c in zip(x_row, y_column)])
                y_column.clear()
        return product

    @classmethod
    def sum(cls, x):
        temp=Tensor(0)
        for tensor in x:
            temp+=tensor
        return temp

    