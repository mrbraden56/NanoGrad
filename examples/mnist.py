import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from nano_grad.engine.matrix import Matrix
from nano_grad.nn.linear import Linear
from nano_grad.nn.network import Network
from nano_grad.nn.optimizer import SGD
from nano_grad.nn.loss import MSE
from nano_grad.nn.activation import ReLU
from nano_grad.nn.activation import Softmax
import pandas as pd
import numpy as np

class FeedForward(Network):
    def __init__(self) -> None:

        self.l1=Linear(784, 64)
        self.l2=Linear(64, 64)
        self.l3=Linear(64, 64)
        self.l4=Linear(64, 10)

        Network.__init__(self, vars(locals()['self']))

    def forward(self, x):
        x1=ReLU(self.l1(x))
        x2=ReLU(self.l2(x1))
        x3=ReLU(self.l3(x2))
        x4=Softmax(self.l4(x3))
        return x4


def main():
    data=pd.read_csv('C:/Users/brade/Research/nano_grad/examples/data/train.csv')
    data=data.loc[:1000,:]
    data = np.array(data)
    dev_data = data[800:].T
    y_test = dev_data[0] # labels
    x_test = dev_data[1:] # pixels
    x_test = x_test / 255. # normalizing pixels

    #separate train data
    train_data = data[:800].T
    y_train = train_data[0] # labels
    x_train = train_data[1:] # pixels
    x_train = x_train / 255.

    x_train=Matrix.array(x_train)
    y_train=Matrix.array(y_train)
    x_test=Matrix.array(x_test)
    y_test=Matrix.array(y_test)
    nn=FeedForward()
    optimizer=SGD(params=nn.parameters(), lr=0.1)
    test=nn.forward(x_train)
    print(test.shape)
    # x=Matrix.array([
    #     [2.0, 3.0, -1.0],
    #     [1.0, 1.0, -1.0],
    #     [-1.0, 2.0, -3.0],
    #     [1.0, 2.0, 3.0]
    # ])
    # ytarget=Matrix.array([1.0, -1.0, 1.0, 1.0])
    # nn=FeedForward()
    # optimizer=SGD(params=nn.parameters(), lr=0.1)
    # for i in range(1000):
    #     ypred=nn.forward(x)
    #     ypred=Matrix.squeeze(ypred, 1)
    #     loss=MSE(ypred=ypred, ytarget=ytarget)
    #     optimizer.zero_grad()
    #     loss.backwards()
    #     optimizer.step()
    #     print(i, loss.data)
    # print(ypred)

if __name__ == "__main__":
    main()

