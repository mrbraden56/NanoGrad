import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from nano_grad.engine.tensor import Tensor
from nano_grad.engine.matrix import Matrix
from nano_grad.nn.linear import Linear

class FeedForward:
    def __init__(self) -> None:
        self.l1=Linear(3, 4)
        self.l2=Linear(4, 4)
        self.l3=Linear(4, 1)
        self.net=[self.l1, self.l2, self.l3]

    def count(self):
        return sum(weights.size() for weights in self.net)

    def parameters(self):
        return [weights.parameters() for weights in self.net]

    def forward(self, x):
        x1=self.l1(x)
        x2=self.l2(x1)
        x3=self.l3(x2)
        return x3


def main():
    x=Matrix.array([
        [2.0, 3.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 2.0, -3.0],
        [1.0, 2.0, 3.0]
    ])
    ytarget=Matrix.array([1.0, -1.0, 1.0, 1.0])
    nn=FeedForward()
    ypred=nn.forward(x)
    print(ypred)

if __name__ == "__main__":
    main()

