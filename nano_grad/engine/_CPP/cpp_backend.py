import ctypes
from ctypes import py_object
import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.ctypeslib import as_array



class CPP:
    def __init__(self)->None:
        self.dispatcher_lib = ctypes.CDLL('../nano_grad/csrc/cmake/Build/libnano_grad_backend_shared.so')

    #test
    def _dot(self, x: np.ndarray, y: np.ndarray, instance, device):
        x_pointer=x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        shape_list = [x.shape[0], x.shape[1]]
        x_shape_ptr = (ctypes.c_int * len(shape_list))(*shape_list)
        
        y_pointer=y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        shape_list = [y.shape[0], y.shape[1]]
        y_shape_ptr = (ctypes.c_int * len(shape_list))(*shape_list)
        
        #this keeps the reference count of Python above 0 so the garbage collector doesnt
        #consume this object while c++ is using it
        instance_py_object=ctypes.py_object(instance)
        instance_address=int(hex(id(instance)), 16)
        c_type_instance_address=ctypes.c_int(instance_address)

        self.dispatcher_lib.call_receive_dot_product.argtypes = [ctypes.POINTER(ctypes.c_double), 
                                                                ctypes.POINTER(ctypes.c_int), 
                                                                ctypes.POINTER(ctypes.c_double), 
                                                                ctypes.POINTER(ctypes.c_int), 
                                                                ctypes.POINTER(ctypes.c_int)]
        
        self.dispatcher_lib.call_receive_dot_product.restype = ctypes.POINTER(ctypes.c_double)

        pointer = self.dispatcher_lib.call_receive_dot_product(x_pointer, 
                                                        x_shape_ptr, 
                                                        y_pointer, 
                                                        y_shape_ptr,
                                                        ctypes.byref(c_type_instance_address)
                                                        )
        
        result_array = as_array(pointer, shape=(x.shape[0], y.shape[1]))
        return result_array
                                                            

