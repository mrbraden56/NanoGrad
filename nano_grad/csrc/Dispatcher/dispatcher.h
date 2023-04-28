#include <Python.h>
#include <string.h>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include "../Ops/ops.h"
#include "../Tensor/tensor.h"


#ifndef DISPATCHER_H
#define DISPATCHER_H

class Dispatcher {
public:
    Dispatcher(); // constructor
    std::vector<std::shared_ptr<Tensor>> receive_dot_product(std::shared_ptr<double> x_array, 
                                                            int* x_shape, 
                                                            std::shared_ptr<double> y_array, 
                                                            int* y_shape,
                                                            std::unique_ptr<int>& python_object);
    std::vector<std::shared_ptr<Tensor>> wrap(std::shared_ptr<double> x, int* x_shape);

private:
    std::map<PyObject*, Ops> instances;
    std::map<PyObject*, Dispatcher> dispatcher_instances;
    std::vector<std::shared_ptr<Tensor>> _prev_call;
};

#endif