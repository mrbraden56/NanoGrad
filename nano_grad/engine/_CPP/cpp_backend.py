import ctypes
from ctypes import py_object
import numpy as np

class CPP:
    def __init__(self)->None:
        self.dispatcher_lib = ctypes.CDLL('../nano_grad/csrc/cmake/Build/dispatcher_build/libdispatcher.so')


    def _dot(self, x: np.ndarray, y: np.ndarray):
        # value=5
        # self.dispatcher_lib.receive_python_object_wrapper.argtypes = [py_object]
        # self.dispatcher_lib.receive_python_object_wrapper.restype = None
        # self.dispatcher_lib.receive_python_object_wrapper(value)
        x_shp_arr = (ctypes.c_int * len(x.shape))(*(x.shape))
        y_shp_arr = (ctypes.c_int * len(y.shape))(*(y.shape))

        self.dispatcher_lib.call_receive_dot_product_shapes.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.dispatcher_lib.call_receive_dot_product_shapes(x_shp_arr, (len(x.shape)), y_shp_arr, (len(y.shape)))

