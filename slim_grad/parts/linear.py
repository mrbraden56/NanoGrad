from slim_grad.parts.neuron import Neuron
from slim_grad.engine.matrix import Matrix
import random

class Linear:
    def __init__(self, nin, nout) -> None:
        # self.neurons = [Neuron(nin) for _ in range(nout)]
        self.nin=nin
        self.nout=nout
        #TODO: Change to Glorot initialization with a normal distribution. 
        self.weights=Matrix.rand_uniform([nin, nout])
        self.b=Matrix.rand_uniform([1, nout])

    def __call__(self, x):
        return Matrix.add_bias(Matrix.dot(x, self.weights), self.b)

    def __repr__(self) -> int:
        return f"Linear Layer (3, 6)"

    def count(self):
        return self.nin*self.nout

    def parameters(self):
        return self.weights