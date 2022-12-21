import random
import numpy as np
import math


# _foo: used internally(private)
# foo_: used when you want a variable that is a keyword
# _: used as placeholder for when you dont need something(for _ in range(100))
# __bar__: dunder methods
# __bar: name mangling
class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        # if isinstance(data, (int, float)): self.data = data
        # if isinstance(data, np.ndarray): self.data = data
        # if isinstance(data, list): self.data = np.array(data)
        self.data=data
        self.grad=0.0
        self._backward=lambda: None
        self._prev=set(_children)
        self._op=_op
        self.label=label

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx]=val

    def __repr__(self) -> str:
        return f"Tensor(data={self.data})"

    def __add__(self, other: type['Tensor']):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out=Tensor(data=self.data+other.data, _children=(self, other), _op='+')
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        return out

    def __mul__(self, other: type['Tensor']):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out=Tensor(data=self.data*other.data, _children=(self, other), _op='*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out

    def zeroes(self, shape):
        weights = [ [0] * shape[1] for i in range(shape[0]) ]
        return weights

    #returns (row, column)
    def shape(self):
        def dim(a):
            if type(a) != list:
                return []
            return [len(a)] + dim(a[0])   
        return dim(self.data)  

    @classmethod
    def rand_uniform(cls, shape):
        weights=cls.zeroes(cls, [shape[0], shape[1]])
        for i in range(shape[0]):
            for j in range(shape[1]):
                weights[i][j]=random.uniform(-1, 1)
        # weights = [ [random.uniform(-1, 1)] * shape[1] for i in range(shape[0]) ]
        return cls(weights)

    @classmethod
    def dot(cls, x, y):
        if x.shape()[1]!=y.shape()[0]: raise Exception("Incorrect array size for dot product")
        product=cls.zeroes(cls, [x.shape()[0], y.shape()[1]])
        y_column=[]
        for row_idx, x_row in enumerate(x.data):
            for col_idx, j in enumerate(range(y.shape()[1])):#col
                for i in range(y.shape()[0]):#row
                    y_column.append(y[i][j])
                #here we have x row and y columns
                #TODO: We need to get rid of sum right here because since it is not going through
                #__add__ function we are not tracking the gradients
                #Also __mult__ is not being called either since when we loop over list we are not loop over Tensor object
                product[row_idx][col_idx]=sum(r*c for r,c in zip(x_row, y_column))
                y_column.clear()
        return cls(product)
        
    @staticmethod
    def add_bias(x, y):
        for i in range(x.shape()[0]):
            x[i]=[xi+yi for xi,yi in zip(x[i], y.data[0])]
        return x      

    #TODO: Remove
    def tanh(self):
        x=self.data
        t=(math.exp(2*x))/(math.exp(2*x)+1)
        # t=np.tanh(x)
        out=Tensor(data=t, _children=(self, ), _op='tanh')
        def _backward():
            # (1-t**2) is basically other.data
            self.grad+=(1-t**2) * out.grad
        out._backward=_backward
        return out

    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()

    # backpropagation
    # 1. Run the forward pass
    # 2. Run the backward pass(Calculatin gradients)
    # 3. Use optimizer.step to change the weights(gradient descent)
    #     - x+=stepSize*x.grad  
