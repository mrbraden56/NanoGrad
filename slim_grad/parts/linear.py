from slim_grad.parts.neuron import Neuron
from slim_grad.engine.matrix import Matrix
import random

class Linear:
    def __init__(self, nin, nout) -> None:
        # self.neurons = [Neuron(nin) for _ in range(nout)]
        self.weights=Matrix.rand_uniform([nin, nout])
        self.b=Matrix.rand_uniform([1, nout])

    def __call__(self, x):
        return Matrix.add_bias(Matrix.dot(x, self.weights), self.b)

    def __repr__(self) -> str:
        return str(len(self.neurons))