#include <Python.h>
#include <string.h>
#include <iostream>
#include <iostream>
#include <vector>


#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
public:
    Tensor(); // constructor
    double* dot(double* x_array, int* x_shape, double* y_array, int* y_shape);


private:
    int example;
    PyObject* instance;

};

#endif