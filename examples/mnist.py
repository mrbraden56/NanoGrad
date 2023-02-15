import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from nano_grad.engine.matrix import Matrix
from nano_grad.nn.linear import Linear
from nano_grad.nn.network import Network
from nano_grad.nn.optimizer import SGD
from nano_grad.nn.loss import MSE
from nano_grad.nn.activation import ReLU
from nano_grad.nn.activation import Log_Softmax

class FeedForward(Network):
    def __init__(self) -> None:

        self.l1=Linear(3, 4)
        self.l2=Linear(4, 4)
        self.l3=Linear(4, 1)

        Network.__init__(self, vars(locals()['self']))

    def forward(self, x):
        x1=ReLU(self.l1(x))
        x2=ReLU(self.l2(x1))
        x3=Log_Softmax(self.l3(x2))
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
    optimizer=SGD(params=nn.parameters(), lr=0.1)
    for i in range(1000):
        ypred=nn.forward(x)
        ypred=Matrix.squeeze(ypred, 1)
        loss=MSE(ypred=ypred, ytarget=ytarget)
        optimizer.zero_grad()
        loss.backwards()
        optimizer.step()
        print(i, loss.data)
    print(ypred)

if __name__ == "__main__":
    main()

