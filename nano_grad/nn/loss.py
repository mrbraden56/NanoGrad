from nano_grad.engine.matrix import Matrix
from nano_grad.engine.tensor import Tensor

def MSE(ypred, ytarget):
    error=Matrix.subtract(ytarget, ypred)
    squared_er=Matrix.pow(error, 2)
    summed=Matrix.sum(squared_er)
    result=summed/ypred.shape[0]
    return result