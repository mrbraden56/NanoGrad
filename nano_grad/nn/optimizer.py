#TODO Create tests for SGD
from nano_grad.engine.tensor import Tensor

class SGD:
    def __init__(self, params, lr=0.01) -> None:
        self.params=params
        self.lr=lr

    def step(self):
        for layer in self.params:
            params, bias = layer
            bias.data+=(-self.lr) * bias.grad
            for i in range(params.shape[0]):
                for j in range(params.shape[1]):
                    params.data[i][j].data+=(-self.lr) * params.data[i][j].grad


    def zero_grad(self):
        for layer in self.params:
            params, bias = layer
            bias.grad=0
            for i in range(params.shape[0]):
                for j in range(params.shape[1]):
                    params.data[i][j].grad=0.0