from nano_grad.engine.tensor import Tensor
from nano_grad.engine._CPP.cpp_backend import CPP
from typing import List
import numpy as np


class SGD(CPP):
    def __init__(self, params, lr=0.01) -> None:
        self.params: List[np.ndarray]=params
        self.lr=lr

    def step(self):
        self._step(self.params, self.lr)

    def zero_grad(self):
        self._zero_grad(self.params)

    def step_old(self):
        for layer in self.params:
            params, bias = layer
            bias.data+=(-self.lr) * bias.grad
            for i in range(params.shape[0]):
                for j in range(params.shape[1]):
                    params.data[i][j].data+=(-self.lr) * params.data[i][j].grad


    def zero_grad_old(self):
        for layer in self.params:
            params, bias = layer
            bias.grad=0
            for i in range(params.shape[0]):
                for j in range(params.shape[1]):
                    params.data[i][j].grad=0.0