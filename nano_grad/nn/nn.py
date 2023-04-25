from nano_grad.engine.matrix import nanopy
from nano_grad.engine.tensor import Tensor
import numpy as np


class Network:
    def __init__(self, network_instance) -> None:
        self.network_instance=network_instance
        self.parameters=None
        self.device=None

    #TODO: Should weights be outside of inner_function so that they dont change each time?
    def linear(self, nin, nout):
        def inner_function(x: Tensor):
            weights = np.random.rand(x.data.shape[1], nout)
            return Tensor.array(nanopy.dot(x.data, weights, self.network_instance))  # + self.bias
        return inner_function

    def parameters(self):
        return nanopy.parameters()