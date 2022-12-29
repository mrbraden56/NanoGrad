import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from slim_grad.parts.linear import Linear
from slim_grad.parts.neuron import Neuron
from slim_grad.engine.tensor import Tensor
from slim_grad.engine.matrix import Matrix
from slim_grad.engine.optimizer import SGD

class FeedForward:
    def __init__(self) -> None:
        #TODO: If neurons get to big the loss/gradients explode
        self.l1=Linear(3, 4)
        self.l2=Linear(4, 4)
        self.l3=Linear(4, 1)
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
    optimizer=SGD(params=nn.parameters(), lr=0.01)
    ypred=nn.forward(x)
    y=Matrix.array([1.0, -1.0, -1.0, 1.0])
    for i in range(100):
        ypred=nn.forward(x)
        ypred=Matrix.squeeze(ypred, 1)
        loss=Matrix.MSE(ypred=ypred, ytarget=y)
        loss.backwards()
        optimizer.step()
        optimizer.zero_grad()
        print(ypred)





if __name__ == '__main__':
    main()