from nano_grad.engine.matrix import nanopy
from nano_grad.engine.tensor import Tensor
import numpy as np
from typing import List



class Network:
    def __init__(self, network_instance) -> None:
        self.network_instance=network_instance
        self.weights: List[np.ndarray] = []
        self.device=None
        self.c_pointer = nanopy.initialize()



    def linear(self, nin, nout):
        weight = np.random.rand(nin, nout)
        self.weights.append(weight)
        def _linear(x: Tensor):
            return Tensor.array(nanopy.dot(x.data, weight, self.network_instance))  # + self.bias
        return _linear

    def parameters(self):
        return self.weights