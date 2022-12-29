import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from slim_grad.parts.linear import Linear
from slim_grad.parts.neuron import Neuron
from slim_grad.engine.tensor import Tensor
from slim_grad.engine.matrix import Matrix

class FeedForward:
    def __init__(self) -> None:
        self.l1=Linear(3, 16)
        self.l2=Linear(16, 32)
        self.l3=Linear(32, 1)
        self.net=[self.l1, self.l2, self.l3]

    def num_parameters(self):
        return sum(weights.num_parameters() for weights in self.net)

    def parameters(self):
        return [weights.parameters() for weights in self.net]

    def forward(self, x):
        x1=self.l1(x)
        x2=self.l2(x1)
        x3=self.l3(x2)
        return x3

    

def main():
    #(4,3)
    x=Matrix.array([
        [2.0, 3.0, -1.0],
        [2.0, 3.0, -1.0],
        [2.0, 3.0, -1.0],
        [2.0, 3.0, -1.0]
    ])
    nn=FeedForward()
    ypred=nn.forward(x)
    y=Matrix.array([1.0, -1.0, -1.0, 1.0])
    ypred=Matrix.squeeze(ypred, 1)
    loss=Matrix.MSE(ypred=ypred, ytarget=y)
    #TODO: Implemented updating data from gradients
    loss.backwards()
    print(loss)
    print(nn.num_parameters())
    print(nn.parameters()[0].shape)






if __name__ == '__main__':
    main()