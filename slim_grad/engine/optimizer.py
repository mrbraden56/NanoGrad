from slim_grad.engine.tensor import Tensor
#TODO: Add ADAM optimizer
class SGD:
    def __init__(self, params, lr=0.01) -> None:
        self.params=params
        self.lr=lr

    def step(self):
        for layer in self.params:
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    layer.data[i][j].data+=(-self.lr) * layer.data[i][j].grad

    def zero_grad(self):
        for layer in self.params:
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    layer.data[i][j].grad=0.0