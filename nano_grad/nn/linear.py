from nano_grad.engine.matrix import Matrix

#TODO: Create tests for Linear
class Linear:
    def __init__(self, nin, nout) -> None:
        self.nin=nin
        self.nout=nout
        self.weights=Matrix.normal(glorot=True, size=[nin, nout])
        self.bias=Matrix.normal(glorot=True)

    def __call__(self, x):
            return Matrix.dot(x, self.weights) + self.bias

    def __repr__(self) -> str:
         return f"Linear(shape={self.nin}, {self.nout})"

    def size(self):
        return self.nin*self.nout+1

    def parameters(self):
        return self.weights, self.bias