from slim_grad.engine.tensor import Tensor
import random
import numpy as np

#Matrix([tensor(5), tensor(3)])
class Matrix:
    def __init__(self, data, shape=None) -> None:
        self.data=data
        self.shape=shape

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
        # if 1 in shape: shape=[max(shape)]
        #1 dim array
        if len(shape)==1:
            for i in range(shape[0]):
                data[i]=Tensor(data[i])
        #2 dim array
        else:
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
        else: weights = [ [Tensor(0)] * shape[1] for i in range(shape[0]) ]
        return cls(weights, shape)

    @classmethod
    def normal(cls, shape):
        weights=cls.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                weights[i][j]=Tensor(random.uniform(-1, 1))
        return weights

    @classmethod
    def glorot_normal(cls, shape):
        #loc->mean, scale->std, 
        # random.normal(loc=0.0, scale=1.0, size=None)
        mean=0
        std=np.sqrt(2/(shape[0]+shape[1]))
        weights=cls.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                weights[i][j]=Tensor(np.random.normal(loc=mean, scale=std))
        return weights

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

    @staticmethod
    def add_bias(x, y):
        for i in range(x.shape[0]):
            x[i]=[xi+yi for xi,yi in zip(x[i], y.data[0])]
        return x 

    #I think we are backpropogating through
    @staticmethod
    def relu(x):
        for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x.data[i][j]=Tensor(max(0, x.data[i][j].data), _children=x.data[i][j]._prev)
        return x

    @classmethod
    def sum(cls, x):
        temp=Tensor(0)
        for tensor in x:
            temp+=tensor
        return temp

    #element_wise
    @classmethod
    def ew_subtract(cls, x, y):
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
    def ew_pow(cls, x, power):
        if len(x.shape)==2:
            product=cls.zeros([x.shape[0], x.shape[1]])
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    product[i][j]=x[i][j]**power
        else:
            product=cls.zeros([x.shape[0]])
            for i in range(x.shape[0]):
                product[i]=x[i] ** power
        return product

    @classmethod
    def MSE(cls, ypred, ytarget):
        error=cls.ew_subtract(ytarget, ypred)
        squared_er=cls.ew_pow(error, 2)
        summed=cls.sum(squared_er)
        result=summed/ypred.shape[0]
        return result

    @classmethod
    def transpose(cls, x):
        result=cls.zeros((x.shape[1], x.shape[0]))
        for j, row in zip(range(result.shape[1]), x):
            for i in range(result.shape[0]):
                result[i][j]=row[i]
        return result

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


                    
            
