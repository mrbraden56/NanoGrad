from nano_grad.engine.matrix import Matrix
from nano_grad.engine.tensor import Tensor

def relu(x):
        for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x.data[i][j]=Tensor(max(0, x.data[i][j].data), _children=x.data[i][j]._prev)
        return x

def MSE(ypred, ytarget):
    error=Matrix.subtract(ytarget, ypred)
    squared_er=Matrix.pow(error, 2)
    summed=Matrix.sum(squared_er)
    result=summed/ypred.shape[0]
    return result