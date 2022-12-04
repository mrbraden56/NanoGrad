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
        self.data=data
        self.grad=0.0
        self._backward=lambda: None
        self._prev=set(_children)
        self._op=_op
        self.label=label

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

    def tanh(self):
        x=self.data
        t=(math.exp(2*x))/(math.exp(2*x)+1)
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

    #TODO: Make everything in terms of numpy and not python lists

