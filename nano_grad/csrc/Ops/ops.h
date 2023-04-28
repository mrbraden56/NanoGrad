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
    std::vector<std::shared_ptr<Tensor>> dot(std::vector<std::shared_ptr<Tensor>> x_array, int* x_shape, 
                             std::vector<std::shared_ptr<Tensor>> y_array, int* y_shape);


private:
    PyObject* _instance;
};

#endif