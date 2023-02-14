import unittest
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from nano_grad.engine.tensor import Tensor
from nano_grad.engine.matrix import Matrix
from nano_grad.nn.linear import Linear

class FeedForward:
    def __init__(self) -> None:
        self.l1=Linear(3, 4)
        self.l2=Linear(4, 4)
        self.l3=Linear(4, 1)
        self.net=[self.l1, self.l2, self.l3]

    def size(self):
        return sum(weights.size() for weights in self.net)

    def parameters(self):
        return [weights.parameters() for weights in self.net]

    def shape(self):
        return [weights.shape() for weights in self.net]

    def forward(self, x):
        x1=self.l1(x)
        x2=self.l2(x1)
        x3=self.l3(x2)
        return x3

class TestTensor(unittest.TestCase):

    #runs once at the begenning
    @classmethod
    def setUpClass(cls):
        cls.a=Matrix.normal(size=[1, 4], glorot=True)
        cls.b=Matrix.normal(size=[2, 6], glorot=True)
        cls.c=Matrix.normal(size=[8, 4], glorot=True)
        cls.d=Matrix.normal(size=[3, 1], glorot=True)

    #runs once at the end
    @classmethod
    def tearDownClass(cls):
        pass

    #runs before each test
    def setUp(self) -> None:
        return super().setUp()

    #runs after each test
    def tearDown(self) -> None:
        return super().tearDown()

    def test_matrix_shape(self):
        l1=Linear(4, 1)
        l2=Linear(6, 5)
        l3=Linear(4, 8)
        l4=Linear(1, 5)
        output1=l1(self.a)
        output2=l2(self.b)
        output3=l3(self.c)
        output4=l4(self.d)
        with self.subTest():
            self.assertEqual(output1.shape, [1, 1])
        with self.subTest():
            self.assertEqual(output2.shape, [2, 5])
        with self.subTest():
            self.assertEqual(output3.shape, [8, 8])
        with self.subTest():
            self.assertEqual(output4.shape, [3, 5])

    def test_count(self):
        nn=FeedForward()
        size=0
        for weight in nn.shape():
            size+=weight[0]*weight[1]+1
        with self.subTest():
            self.assertEqual(size, nn.size())


if __name__ == '__main__':
    unittest.main()