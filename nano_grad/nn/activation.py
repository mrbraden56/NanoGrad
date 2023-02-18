from nano_grad.engine.tensor import Tensor
from nano_grad.engine.matrix import Matrix
import numpy as np

def ReLU(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x.data[i][j]=x.data[i][j].relu()
    return x

# e_x = np.exp(x - np.max(x))
# softmax = e_x / e_x.sum()
# log_softmax = np.log(softmax)
def Softmax(x):
    prob=[]
    averaged=x-Matrix.max(x)
    e_x=Matrix.exp(averaged)
    summed__e_x= Matrix.sum(e_x)
    softmax=e_x / summed__e_x
    return softmax