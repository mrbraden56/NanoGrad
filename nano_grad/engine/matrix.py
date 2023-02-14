import os
from nano_grad.engine.tensor import Tensor
import random
import numpy as np

#TODO: See if there is anything we can do better in terms of design
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

    def __add__(self, other: type['Matrix']):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.data[i][j]+=other.data
        return self

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
            weights = [ [Tensor(0)] * shape[1] for i in range(shape[0]) ]
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

    @classmethod
    def normal(cls, size=None, glorot=False):
        if size==None: return Tensor(random.uniform(-1, 1)) 
        weights=cls.zeros(size)
        mean=0
        std=np.sqrt(2/(size[0]+size[1]))
        for i in range(size[0]):
            for j in range(size[1]):
                if not glorot: weights[i][j]=Tensor(random.uniform(-1, 1))
                if glorot: weights[i][j]=Tensor(np.random.normal(loc=mean, scale=std))
        return weights

    @classmethod
    def subtract(cls, x, y, element_wise=True):
        pass

    @classmethod
    def pow(cls, x, y, element_wise=True):
        pass

    