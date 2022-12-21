from slim_grad.engine.tensor import Tensor
import random
import numpy as np

class Neuron:
    def __init__(self, nin) -> None:
        #act=dot(a, b.T)+b
        # self.w=[Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.w=Tensor.weight_matrix(x.shape)
        self.b=Tensor(random.uniform(-1,1))

    # n=Neuron(2)
    # n(x)
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out=act.tanh()
        return out