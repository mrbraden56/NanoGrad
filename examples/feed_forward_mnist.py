import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from slim_grad.parts.linear import Linear
from slim_grad.parts.neuron import Neuron
from slim_grad.engine.tensor import Tensor

class FeedForward:
    def __init__(self) -> None:
        self.l1=Linear(64, 128)
        self.l2=Linear(128, 256)
        self.l3=Linear(256, 128)
        self.l4=Linear(128, 64)
        self.l5=Linear(64, 1)

    def forward(self, x):
        x1=self.l1(x)
        print(len(x1))

    

#TODO: Make everything a custom tensor array
def main():
    # x=np.array([1, 2, 3, 4])
    # nn=FeedForward()
    # nn.forward(x)
    x=Tensor([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ])
    print(x.shape())
    weights=Tensor.weight_matrix([3,4])
    print(weights.shape())
    # y=[1.0, -1.0, -1.0, 1.0]
    # linear_1=Linear(3, 6)
    # out=linear_1(x)
    # print(out, type(out))





if __name__ == '__main__':
    main()