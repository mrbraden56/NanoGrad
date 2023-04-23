#include <Python.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <memory>
#include "../Tensor/tensor.h"


#ifndef OPS_H
#define OPS_H

class Ops {
public:
    Ops(); // constructor
    std::vector<Tensor*> dot(std::vector<Tensor*> x_array, int* x_shape, 
                             std::vector<Tensor*> y_array, int* y_shape);


private:
    PyObject* _instance;
};

#endif