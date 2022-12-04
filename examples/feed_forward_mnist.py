import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from slim_grad.parts.linear import Linear

class FeedForward:
    def __init__(self) -> None:
        self.l1=Linear(64, 128)
        self.l2=Linear(128, 256)
        self.l3=Linear(256, 128)
        self.l4=Linear(128, 64)
        self.l5=Linear(64, 1)

    


def main():
    nn=FeedForward()
    print(nn.l1)


if __name__ == '__main__':
    main()