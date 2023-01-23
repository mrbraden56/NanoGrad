import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from slim_grad.engine.tensor import Tensor
from slim_grad.engine.matrix import Matrix

#TODO: Add unittests to make sure backprop is working
#   like keeping track of _prev, correct graph, etc