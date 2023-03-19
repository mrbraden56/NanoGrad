from nano_grad.engine.matrix import nanopy
import numpy as np


class Network:
    def __init__(self, network_instance) -> None:
        self.network_instance=network_instance
        self.parameters=None
        self.device=None

    def linear(self, nin, nout):
        def inner_function(x):
            weights = np.random.rand(x.shape[1], nout)
            return nanopy.dot(x, weights, self.network_instance)  # + self.bias
        return inner_function

