from slim_grad.parts.neuron import Neuron
from slim_grad.engine.tensor import Tensor
import random

class Linear:
    def __init__(self, nin, nout) -> None:
        # self.neurons = [Neuron(nin) for _ in range(nout)]
        self.weights=Tensor.rand_uniform([nin, nout])
        self.b=Tensor.rand_uniform([1, nout])

    def __call__(self, x):
        return Tensor.add_bias(Tensor.dot(x, self.weights), self.b)

    def __repr__(self) -> str:
        return str(len(self.neurons))