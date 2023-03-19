import ctypes
from ctypes import py_object
import numpy as np

class CPP:
    def __init__(self)->None:
        self.dispatcher_lib = ctypes.CDLL('../nano_grad/csrc/cmake/Build/dispatcher_build/libdispatcher.so')


    def _dot(self, x: np.ndarray, y: np.ndarray, instance, device):
        x_shp_arr = (ctypes.c_int * len(x.shape))(*(x.shape))
        y_shp_arr = (ctypes.c_int * len(y.shape))(*(y.shape))
        #this keeps the reference count of Python above 0 so the garbage collector doesnt
        #consume this object while c++ is using it
        instance_py_object=ctypes.py_object(instance)
        instance_address=int(hex(id(instance)), 16)
        c_type_instance_address=ctypes.c_int(instance_address)

        self.dispatcher_lib.call_receive_dot_product_shapes.argtypes = [ctypes.POINTER(ctypes.c_int), 
                                                                        ctypes.c_int, ctypes.POINTER(ctypes.c_int), 
                                                                        ctypes.c_int, 
                                                                        ctypes.POINTER(ctypes.c_int)]
        self.dispatcher_lib.call_receive_dot_product_shapes(x_shp_arr, 
                                                            (len(x.shape)), 
                                                            y_shp_arr, 
                                                            (len(y.shape)),
                                                            ctypes.byref(c_type_instance_address)
                                                            )
                                                            

