import unittest
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from nano_grad.engine.tensor import Tensor


class TestTensor(unittest.TestCase):

    #runs once at the begenning
    @classmethod
    def setUpClass(cls):
        cls.x=[1, 4, 2, 6, 9, 12.3, -5, 23.3, 574.2]
        cls.y=[-3, 6, 1, 4.5, 6, 3.7, -900, 11, 42]

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

    def test_addition_gradients(self):
        for x,y in zip(self.x, self.y):
            a=Tensor(data=x, _op="+", label="a")
            b=Tensor(data=y, _op="+", label="b")
            c=a+b
            c.backwards()
            with self.subTest():
                self.assertEqual(a.grad, 1)
            with self.subTest():
                self.assertEqual(b.grad, 1)
            with self.subTest():
                self.assertEqual(a.data, x)
            with self.subTest():
                self.assertEqual(b.data, y)
            with self.subTest():
                self.assertEqual(c.data, a.data+b.data)

    def test_addition_children(self):
        for x,y in zip(self.x, self.y):
            a=Tensor(data=x, _op="+", label="a")
            b=Tensor(data=y, _op="+", label="b")
            c=a+b
            children=[]
            children.append(c._prev.pop().data)
            children.append(c._prev.pop().data)
            with self.subTest():
                self.assertIn(x, children)
            with self.subTest():
                self.assertIn(y, children)

    def test_multiplication_gradients(self):
        for x,y in zip(self.x, self.y):
            d=Tensor(data=x, _op="*", label="a")
            e=Tensor(data=y, _op="*", label="b")
            f=d*e
            f.backwards()
            with self.subTest():
                self.assertEqual(d.grad, e.data)
            with self.subTest():
                self.assertEqual(e.grad, d.data)
            with self.subTest():
                self.assertEqual(d.data, x)
            with self.subTest():
                self.assertEqual(e.data, y)
            with self.subTest():
                self.assertEqual(f.data, d.data*e.data)

    def test_multiplication_children(self):
        for x,y in zip(self.x, self.y):
            d=Tensor(data=x, _op="*", label="a")
            e=Tensor(data=y, _op="*", label="b")
            f=d*e
            children=[]
            children.append(f._prev.pop().data)
            children.append(f._prev.pop().data)
            with self.subTest():
                self.assertIn(x, children)
            with self.subTest():
                self.assertIn(y, children)

    def test_power_gradients(self):
        for x,y in zip(self.x, self.y):
            d=Tensor(data=x, _op="**", label="a")
            e=Tensor(data=y, _op="**", label="b")
            f=d**e
            f.backwards()
            with self.subTest():
                self.assertEqual(d.grad, (e*(d**(e-1))).data)
            with self.subTest():
                if x<1: self.assertEqual(e.grad, 0)
                else: self.assertEqual(e.grad, np.log(x)*(x**y))
            with self.subTest():
                self.assertEqual(d.data, x)
            with self.subTest():
                self.assertEqual(e.data, y)
            with self.subTest():
                self.assertEqual(f.data, x**y)

    def test_power_children(self):
        for x,y in zip(self.x, self.y):
            d=Tensor(data=x, _op="**", label="a")
            e=Tensor(data=y, _op="**", label="b")
            f=d**e
            children=[]
            children.append(f._prev.pop().data)
            children.append(f._prev.pop().data)
            with self.subTest():
                self.assertIn(x, children)
            with self.subTest():
                self.assertIn(y, children)

    #TODO: Test method for correct topo graph

    

if __name__ == '__main__':
    unittest.main()