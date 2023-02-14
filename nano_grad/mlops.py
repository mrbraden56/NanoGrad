from nano_grad.engine.matrix import Matrix

#TODO Create ReLU

#TODO Create MSE
def MSE(ypred, ytarget):
    error=Matrix.ew_subtract(ytarget, ypred, element_wise=True)
    squared_er=Matrix.ew_pow(error, 2, element_wise=True)
    summed=Matrix.sum(squared_er)
    result=summed/ypred.shape[0]
    return result