#include <Python.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <memory>


#ifndef OPS_H
#define OPS_H

class Ops {
public:
    Ops(); // constructor
    double* dot(double* x_array, int* x_shape, double* y_array, int* y_shape);


private:
    PyObject* _instance;
};

#endif