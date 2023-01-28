import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from slim_grad.engine.tensor import Tensor
from slim_grad.engine.matrix import Matrix

#TODO: Add unittests to make sure backprop is working
#   like keeping track of _prev, correct graph, etc

class TestTensor(unittest.TestCase):

    #runs once at the begenning
    @classmethod
    def setUpClass(cls):
        cls.x=1

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

    #3 nodes with 2 inputs and 1 ouput
    def test_addition_gradients(self):
        a=Tensor(data=3, _op="+", label="a")
        b=Tensor(data=2, _op="+", label="b")
        c=a+b
        c.backwards()
        with self.subTest():
            self.assertEqual(a.grad, 1)
        with self.subTest():
            self.assertEqual(b.grad, 1)

    #3 nodes with 2 inputs and 1 ouput
    def test_multiplication_gradients(self):
        a=Tensor(data=3, _op="*", label="a")
        b=Tensor(data=2, _op="*", label="b")
        c=a*b
        c.backwards()
        with self.subTest():
            self.assertEqual(a.grad, 2)
        with self.subTest():
            self.assertEqual(b.grad, 3)

    #3 nodes with 2 inputs and 1 ouput
    #TODO: work this out by hand and make sure it is right
    def test_power_gradients(self):
        a=Tensor(data=3, _op="*", label="a")
        b=Tensor(data=2, _op="*", label="b")
        c=b**a
        c.backwards()
        with self.subTest():
            self.assertEqual(a.grad, 2)
        with self.subTest():
            self.assertEqual(b.grad, 3)

    #make suregraph has correct ordering
    def test_correct_topo_graph(self):
        pass

    #make sure nodes have correct parents and children
    def test_prev_nodes(self):
        pass

    #test gradients pass through ReLU correctly
    def test_pass_through_activation(self):
        pass

if __name__ == '__main__':
    unittest.main()



