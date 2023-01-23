import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from slim_grad.engine.tensor import Tensor
from slim_grad.engine.matrix import Matrix

class TestMatrix(unittest.TestCase):

    #runs once at the begenning
    @classmethod
    def setUpClass(cls):
        cls.test_x=[
            #(4,3)
            Matrix.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0]
            ]),
            #(4,1)
            Matrix.array([
                [1.0],
                [2.0],
                [3.0],
                [4.0]
            ]),
            #(3,4)
            Matrix.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 6.0],
            ]),
        ]

        cls.test_y=[
            #(3,4)
            Matrix.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 6.0],
            ]),
            #(1,4)
            Matrix.array([
                [1.0, 2.0, 3.0, 4.0]
            ]),
            #(4,3)
            Matrix.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0]
            ]),
        ]

        cls.test_npx=[
            np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0]
            ]),
            np.array([
                [1.0],
                [2.0],
                [3.0],
                [4.0]
            ]),
            np.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 6.0],
            ]),
        ]

        cls.test_npy=[
            np.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 6.0],
            ]),
            np.array([
                [1.0, 2.0, 3.0, 4.0]
            ]),
            np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0]
            ]),
        ]

        cls.shapes=[
            (1,4),
            (1,1),
            (5,3),
        ]

        cls.sums=[
            [1, 2, 3, 4, 5],
            [-1, -3, -4, 5],
            [3.0, 2.6, 5]
        ]

        cls.test_ewx=[
            #(4,3)
            Matrix.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0]
            ]),
            #(4,1)
            Matrix.array([
                [1.0],
                [2.0],
                [3.0],
                [4.0]
            ]),
            #(3,4)
            Matrix.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 6.0],
            ]),
        ]

        cls.test_ewy=[
            #(4,3)
            Matrix.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0]
            ]),
            #(4,1)
            Matrix.array([
                [1.0],
                [2.0],
                [3.0],
                [4.0]
            ]),
            #(3,4)
            Matrix.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 6.0],
            ]),
        ]

        cls.test_npewx=[
            np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0]
            ]),
            np.array([
                [1.0],
                [2.0],
                [3.0],
                [4.0]
            ]),
            np.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 6.0],
            ]),
        ]

        cls.test_npewy=[
            np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0]
            ]),
            np.array([
                [1.0],
                [2.0],
                [3.0],
                [4.0]
            ]),
            np.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 6.0],
            ]),
        ]

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

    def test_dot(self):
        for x, y, npx, npy in zip(self.test_x, self.test_y, self.test_npx, self.test_npy):
            np_dot_product=np.dot(npx, npy)
            dot_product=Matrix.dot(x, y)
            self.assertEqual(x.shape[0], y.shape[1])
            for i in range(dot_product.shape[0]):
                for j in range(dot_product.shape[1]):
                    self.assertEqual(dot_product[i][j].data, np_dot_product[i][j])

    def test_zeros(self):
        for shape in self.shapes:
            np_zeros=np.zeros(shape)
            zeros=Matrix.zeros(shape)
            self.assertEqual(zeros.shape, np_zeros.shape)
            for i in range(zeros.shape[0]):
                for j in range(zeros.shape[1]):
                    self.assertEqual(zeros[i][j].data, np_zeros[i][j])

    def test_rand_uniform(self):
        for shape in self.shapes:
            np_rand=np.zeros(shape)
            rand=Matrix.zeros(shape)
            self.assertEqual(rand.shape, np_rand.shape)

    def test_sum(self):
        for list_ in self.sums:
            self.assertEqual(Matrix.sum(list_).data, np.sum(list_))

    def test_ewsubtract(self):
        for x, y, npx, npy in zip(self.test_ewx, self.test_ewy, self.test_npewx, self.test_npewy):
            np_ew_subtract=np.subtract(npx, npy)
            ew_subtract=Matrix.ew_subtract(x,y)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    self.assertEqual(ew_subtract[i][j].data, np_ew_subtract[i][j])

    def test_ewpow(self):
        for x, npx in zip(self.test_ewx, self.test_npewx):
            np_ew_pow=np.power(npx, 2)
            ew_pow=Matrix.ew_pow(x, 2)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    self.assertEqual(ew_pow[i][j].data, np_ew_pow[i][j])



if __name__ == '__main__':
    unittest.main()