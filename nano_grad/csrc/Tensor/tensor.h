#include <Python.h>
#include <string.h>
#include <iostream>
#include <functional>
#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <memory>

#ifndef TENSOR_H
#define TENSOR_H

class Tensor{
    public:
        Tensor(std::shared_ptr<double> data, double grad, std::function<void()>& _backward, std::vector<Tensor>& _prev);
        void test_parents(double parent1, double parent2);
        int depth() const;
        void backwards();

        int* shape;
        std::shared_ptr<double> data;
        double grad;
        std::function<void()> _backward;
        std::vector<Tensor> _prev;
        std::vector<std::vector<Tensor>> parameters;

        Tensor operator+(const Tensor& other);
        Tensor operator*(const Tensor& other);

    private:

};

#endif