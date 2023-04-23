#include <Python.h>
#include <string.h>
#include <iostream>
#include <functional>
#include <vector>

#ifndef TENSOR_H
#define TENSOR_H

class Tensor{
    public:
        Tensor(double* data, double grad, std::function<void()>& _backward, std::vector<Tensor>& _prev);
        int* shape;
        double* data;
        double grad;
        std::function<void()> _backward;
        std::vector<Tensor> _prev;

        Tensor operator+(const Tensor& other);
        Tensor operator*(const Tensor& other);

    private:

};

#endif