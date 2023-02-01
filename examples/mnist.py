import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from nano_grad.engine.tensor import Tensor
from nano_grad.engine.matrix import Matrix

def main():
    x=Matrix.array([
        [2.0, 3.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 2.0, -3.0],
        [1.0, 2.0, 3.0]
    ])
    y=Matrix.array([
        [2.0, 3.0, -1.0, 1],
        [1.0, 1.0, -1.0, 2],
        [-1.0, 2.0, -3.0, 5],
    ])
    z=Matrix.dot(x,y)
    print(z)

if __name__ == "__main__":
    main()

