#include <Python.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <memory>


#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
public:
    Tensor(); // constructor
    double* dot(double* x_array, int* x_shape, double* y_array, int* y_shape);


private:
    int example;
    PyObject* instance;
    void MyGemm(int m, int n, int k,
	     double *A, int ldA,
	     double *B, int ldB,
	     double *C, int ldC );
    void MyGemv(int m, int n, double *A, int ldA,
           double *x, int incx, double *y, int incy);
    void Axpy(int n, double alpha, double *x, int incx, double *y, int incy);

};

#endif