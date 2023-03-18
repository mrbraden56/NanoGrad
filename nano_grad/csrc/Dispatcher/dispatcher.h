#include <Python.h>
#include <string.h>
#include <iostream>
#include <iostream>
#include <vector>


#ifndef DISPATCHER_H
#define DISPATCHER_H

class Dispatcher {
public:
    Dispatcher(); // constructor
    void printMessage(); // member function
    void receive_dot_product_shapes(int* x_shape, int x_shape_length, int* y_shape, int y_shape_length);

private:
    int example;
};

#endif