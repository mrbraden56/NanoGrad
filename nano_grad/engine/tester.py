import os
from ctypes import cdll
import ctypes

# Get the full path to the directory containing the Python script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the shared library using the constructor syntax for the full path
lib = ctypes.CDLL('/mnt/c/Users/brade/Research/nano_grad/nano_grad/engine/tester.so')
class Foo(object):
    def __init__(self):
        self.obj = lib.Foo_new()

    def bar(self):
        lib.Foo_bar(self.obj)
# Once you have that you can call it like

f = Foo()
f.bar() #and you will see "Hello" on the screen
