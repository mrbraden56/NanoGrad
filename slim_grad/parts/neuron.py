from slim_grad.engine.tensor import Tensor
import random

class Neuron:
    def __init__(self, nin) -> None:
        self.w=[Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b=Tensor(random.uniform(-1,1))

    # n=Neuron(2)
    # n(x)
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out=act.tanh()
        return out