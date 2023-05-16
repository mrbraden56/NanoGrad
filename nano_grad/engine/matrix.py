import os
from nano_grad.engine.tensor import Tensor
from nano_grad.engine._CPP.cpp_backend import CPP
import random
import numpy as np




class nanopy(CPP):
    """

    Intermediate class to communicate with backend

    """
    def __init__(self)->None:
        super().__init__()
        pass
    
    @classmethod
    def dot(cls, x: np.ndarray, y: np.ndarray, network_instance=None, device="cpu"):
        return cls()._dot(x, y, network_instance, device)
    
    @classmethod
    def parameters(cls, network_instance=None, device="cpu"):
        return cls()._parameters(network_instance, device)

    @classmethod
    def initialize(cls):
        return cls()._initialize()









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

    def __sub__(self, other: type['Matrix']):
        other = other if isinstance(other, Tensor) else Tensor(other)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.data[i][j]-=other.data
        return self

    def __pow__(self, other: type['Matrix']):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.data[i][j]=self.data[i][j]**other.data
        return self
    
    def __truediv__(self, other: type['Tensor']):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.data[i][j]=self.data[i][j]/other
        return self

    @classmethod
    def array(cls, data):
        def get_shape(lst):
            if not isinstance(lst, list):
                return []
            shape = [len(lst)]
            subshape = get_shape(lst[0])
            return shape + subshape
        if isinstance(data, np.ndarray):
            shape = data.shape
            if len(shape)==1: product=cls.zeros([data.shape[0], 1])
            else: product=cls.zeros([data.shape[0], data.shape[1]])
        else:
            shape= get_shape(data)
        if len(shape)==1:
            for i in range(shape[0]):
                if isinstance(data[i], Tensor):
                    return cls(data, shape)
                elif isinstance(data, np.ndarray):
                    product[i][0]=int(data[i].astype(int))
                else: data[i]=Tensor(data[i])
        elif len(shape)==2:
            for i in range(shape[0]):
                if len(shape)==1: 
                    data[i]=Tensor(data[i])
                    continue
                for j in range(shape[1]):
                    if isinstance(data, np.ndarray):
                        number=int(data[i][j].astype(int))
                        product[i][j]=Tensor(number)
                    else: data[i][j]=Tensor(data[i][j])
        if isinstance(data, np.ndarray): return product
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
    def dot(cls, x, y, backend):
        if not backend:
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
        if backend:
            print(x.shape, y.shape)

    @classmethod
    def sum(cls, x):
        #list, Matrix
        x = x if isinstance(x, Matrix) else cls.array(x)
        temp=Tensor(0)
        if len(x.shape)==2:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    test=x[i][j]
                    temp+=x[i][j]
        else:
            for i in range(x.shape[0]):
                temp+=x[i]
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
    def subtract(cls, x, y):
        if tuple(x.shape)!=tuple(y.shape): raise Exception("Matrices not of same size")
        result=cls.zeros(x.shape)
        if len(x.shape)==2:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    result[i][j]=x[i][j]-y[i][j]
        else:
            for i in range(x.shape[0]):
                result[i]=x[i] - y[i]
        return result
    
    @classmethod
    def divide(cls, x, y):
        if tuple(x.shape)!=tuple(y.shape): raise Exception("Matrices not of same size")
        result=cls.zeros(x.shape)
        if len(x.shape)==2:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    result[i][j]=x[i][j] * (1 / y[i][j])
        else:
            for i in range(x.shape[0]):
                result[i]=x[i] * (1 / y[i])
        return result

    @classmethod
    def pow(cls, x, y):
        if len(x.shape)==2:
            product=cls.zeros([x.shape[0], x.shape[1]])
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    product[i][j]=x[i][j]**y
        else:
            product=cls.zeros([x.shape[0]])
            for i in range(x.shape[0]):
                product[i]=x[i] ** y
        return product

    @classmethod
    def squeeze(cls, x, dim):
        #only works for 2 dims
        new_shape=x.shape[int(not bool(dim))]
        result=cls.zeros([new_shape])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x.shape[0]==1:
                    result[j]==x[i][j]
                elif x.shape[1]==1:
                    result[i]=x[i][j]
        return result
    
    @classmethod
    def exp(cls, x):
        e=Tensor(2.718281)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i][j]=e**x[i][j]
        return x
    
    @staticmethod
    def max(x):
        max_tracker=x[0][0].data
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j].data>max_tracker:
                    max_tracker=x[i][j].data
        return max_tracker
    
    @classmethod
    def transpose(cls, x):
        result=cls.zeros((x.shape[1], x.shape[0]))
        for j, row in zip(range(result.shape[1]), x):
            for i in range(result.shape[0]):
                result[i][j]=row[i]
        return result

    