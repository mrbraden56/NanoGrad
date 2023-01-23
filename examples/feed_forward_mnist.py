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
        self.l1=Linear(3, 8)
        self.l2=Linear(8, 8)
        self.l3=Linear(8, 4)
        self.l4=Linear(4, 1)
        self.net=[self.l1, self.l2, self.l3, self.l4]

    def count(self):
        return sum(weights.count() for weights in self.net)

    def parameters(self):
        return [weights.parameters() for weights in self.net]

    def forward(self, x):
        x1=Matrix.relu(self.l1(x))
        x2=Matrix.relu(self.l2(x1))
        x3=Matrix.relu(self.l3(x2))
        x4=self.l4(x3)
        return x4

    

def main():
    x=Matrix.array([
        [2.0, 3.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 2.0, -3.0],
        [1.0, 2.0, 3.0]
    ])
    nn=FeedForward()
    print(nn.count())
    optimizer=SGD(params=nn.parameters(), lr=0.1)
    y=Matrix.array([2.0, -1.0, -3.0, 1.0])
    #with 9999 epochs we got down to 3.4e-06 loss and perfect prediction of y
    #problem is some times it explodes or doesnt find a local minimum(vanishes) and gets stuck
    #     at a loss that is too high like 2.
    for i in range(1000):
        ypred=nn.forward(x)
        ypred=Matrix.squeeze(ypred, 1)
        loss=Matrix.MSE(ypred=ypred, ytarget=y)
        optimizer.zero_grad()
        loss.backwards()
        optimizer.step()
        print(i, loss.data)
    print(ypred)




if __name__ == '__main__':
    main()