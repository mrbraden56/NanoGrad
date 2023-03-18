from nano_grad.engine.matrix import nanopy
import numpy as np


# class Linear:
#     def __init__(self, nin, nout)->None:
#         self.nin=nin
#         self.nout=nout

#     def __call__(self, x):
#         weights=np.random.rand(x.shape[1], self.nout)
#         print(self.network_instance)
#         return nanopy.dot(x, weights)# + self.bias

class Network:
    def __init__(self, network_instance) -> None:
        self.network_instance=network_instance
        self.parameters=None
        self.device=None

    def linear(self, nin, nout):
        def inner_function(x):
            weights = np.random.rand(x.shape[1], nout)
            return nanopy.dot(x, weights)  # + self.bias
        print(self.network_instance)
        return inner_function

