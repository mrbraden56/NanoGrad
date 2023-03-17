from nano_grad.engine.matrix import Matrix
from nano_grad.engine.matrix import nanopy
import numpy as np

class Linear:
    def __init__(self, nin, nout) -> None:
        self.nin=nin
        self.nout=nout
        # self.weights=Matrix.normal(size=[nin, nout], glorot=True)
        self.weights=np.random.rand(nin, nout)
        self.bias=Matrix.normal(size=None, glorot=True)

    def __call__(self, x):
        return nanopy.dot(x, self.weights)# + self.bias

    def __repr__(self) -> str:
        return f"Linear(shape={self.nin}, {self.nout})"

    def size(self):
        return self.nin*self.nout+1

    def parameters(self):
        return self.weights, self.bias

    def shape(self):
        return [self.nin, self.nout]                                                             