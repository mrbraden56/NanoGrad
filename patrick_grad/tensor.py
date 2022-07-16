import numpy as np


class nn:
    def __init__(self):
        pass

    def Linear(self):
        pass

    def Conv1d(self):
        pass

    def Conv2d(self):
        pass


class Patrick:
    def __init__(self, data):
        if isinstance(data, list):
            self.array = np.array(data)
        elif isinstance(data, np.ndarray):
            self.array = np.array(data)
        else:
            raise Exception(f"Unable to create patrick of type <'{type(data)}'>")
        self.requires_grad = False

    def __repr__(self):
        return f"Patrick ({self.array}, requires_grad={self.requires_grad})"

    @classmethod
    def randn(cls, *shape):
        return cls(np.random.randn(*shape).astype(np.float32))

    @classmethod
    def zeros(cls, *shape):
        return cls(np.zeros(*shape))

    @property
    def shape(self):
        return self.array.shape

    @property
    def get_array(self):
        return self.array

    @classmethod
    def matmul(cls, x1, x2):
        arr1 = x1.get_array()
        arr2 = x2.get_array()
        return cls(np.matmul(arr1, arr2))


def main():
    x = Patrick.zeros((3, 2))
    y = Patrick.zeros((3, 2))
    z = Patrick.matmul(x, y)
    print(x)


if __name__ == '__main__':
    main()
