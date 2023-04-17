#include <Python.h>
#include <string.h>
#include <iostream>
#include <iostream>
#include <vector>
#include <map>
#include "../Ops/ops.h"
#include "../Tensor/tensor.h"


#ifndef DISPATCHER_H
#define DISPATCHER_H

class Dispatcher {
public:
    Dispatcher(); // constructor
    double* receive_dot_product(double* x_array, 
                             int* x_shape, 
                             double* y_array, 
                             int* y_shape,
                             int* python_object);

private:
    std::map<PyObject*, Ops> instances;
};

#endif