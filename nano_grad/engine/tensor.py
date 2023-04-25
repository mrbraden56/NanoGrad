import numpy as np

class Tensor:
    def __init__(self, data, shape=None, _children=(), _op='', label='') -> None:
        self.data=data
        self.shape=shape
        # self._prev=set(_children)
        # self._op=_op
        # self.label=label
        # self.grad=0.0
        # self._backward=lambda: None  

    def __repr__(self) -> str:
        return f"Tensor(data={self.data})"

    @classmethod
    def array(cls, data):
        tensor=cls(data=data, shape=data.shape)
        return tensor

    def relu(self):
        out=Tensor(data=max(0, self.data), _children=(self, ), _op='')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward=_backward
        return out

    def log_softmax(self):
        softmax=np.exp(self.data)
        out=Tensor(data=softmax, _children=(self, ), _op='')
        def _backward():
            pass
        out._backward=_backward
        return out
    
        # def __add__(self, other: type['Tensor']):
    #     other = other if isinstance(other, Tensor) else Tensor(other)
    #     out=Tensor(data=self.data+other.data, _children=(self, other), _op='+')
    #     def _backward():
    #         self.grad+=out.grad
    #         other.grad+=out.grad
    #     out._backward=_backward
    #     return out

    # def __mul__(self, other: type['Tensor']):
    #     other = other if isinstance(other, Tensor) else Tensor(other)
    #     out=Tensor(data=self.data*other.data, _children=(self, other), _op='*')
    #     def _backward():
    #         self.grad+=other.data*out.grad
    #         other.grad+=self.data*out.grad
    #     out._backward=_backward
    #     return out

    # def __pow__(self, other: type['Tensor']):
    #     other = other if isinstance(other, Tensor) else Tensor(other)
    #     out=Tensor(data=self.data**other.data, _children=(self, other), _op='**')
    #     def _backward():
    #         self.grad=(other.data*(self.data**(other.data-1)))*out.grad
    #         if self.data<1: 
    #             other.grad=0
    #         else: 
    #             other.grad=((np.log(self.data))*(self.data**other.data))*out.grad
    #     out._backward=_backward
    #     return out

    # def __neg__(self):
    #     return self * (-1)

    # def __sub__(self, other):
    #     return self + (-other)

    # def __rsub__(self, other):
    #     return other + (-self)

    # def __radd__(self, other):
    #     return other+self

    # def __rmul__(self, other):
    #     return other*self

    # def __truediv__(self, other):
    #     return self * other**-1

    # def __rtruediv__(self, other):
    #     return other * self**-1

    # def backwards(self):
    #     topo=[]
    #     visited=set()
    #     def build_topo(v):
    #         if v not in visited:
    #             visited.add(v)
    #             for child in v._prev:
    #                 build_topo(child)
    #             topo.append(v)
    #     build_topo(self)
    #     self.grad=1.0
    #     for node in reversed(topo):
    #         node._backward()