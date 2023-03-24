#include <Python.h>
#include <string.h>
#include <iostream>
#include <iostream>
#include <vector>
#include <map>
// #include "python_tensor.h"


#ifndef DISPATCHER_H
#define DISPATCHER_H

class Dispatcher {
public:
    Dispatcher(); // constructor
    void printMessage(); // member function
    void receive_dot_product(double* x_array, 
                             int* x_shape, 
                             int* y_shape, 
                             int y_shape_length,
                             int* python_object);

private:
    std::map<PyObject*, std::string> instances;
};

#endif