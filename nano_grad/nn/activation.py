from nano_grad.engine.tensor import Tensor

def ReLU(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x.data[i][j]=x.data[i][j].relu()
    return x

def Log_Softmax(x):
    print(x.data)
    return x.log