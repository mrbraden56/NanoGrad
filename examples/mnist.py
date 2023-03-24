import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from nano_grad.engine.matrix import Matrix
from nano_grad.nn import nn
from nano_grad.nn.optimizer import SGD
from nano_grad.nn.loss import MSE
from nano_grad.nn.activation import ReLU
from nano_grad.nn.activation import Softmax
import pandas as pd
import numpy as np
import ctypes
from ctypes import py_object

class FeedForward(nn.Network):
    def __init__(self) -> None:
        super().__init__(self)
        self.l1 = self.linear(4, 1)

    def forward(self, x):
        x1 = self.l1(x)
        return x1


def main():
    nn=FeedForward()
    x=np.random.rand(4, 3)
    y=np.array([1, -1, 1, 1])
    ypred=nn.forward(x)
    ypred2=nn.forward(x)


    # data=pd.read_csv('C:/Users/brade/Research/nano_grad/examples/data/train.csv')
    # data=data.loc[:1000,:]
    # data = np.array(data)
    # dev_data = data[800:].T
    # y_test = dev_data[0] # labels
    # x_test = dev_data[1:] # pixels
    # x_test = x_test / 255. # normalizing pixels

    # #separate train data
    # train_data = data[:800].T
    # y_train = train_data[0] # labels
    # x_train = train_data[1:] # pixels
    # x_train = x_train / 255.

    # x_train=Matrix.array(x_train)
    # y_train=Matrix.array(y_train)
    # x_test=Matrix.array(x_test)
    # y_test=Matrix.array(y_test)
    # optimizer=SGD(params=nn.parameters(), lr=0.1)
    # test=nn.forward(x_train)
    # print(test.shape)
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

