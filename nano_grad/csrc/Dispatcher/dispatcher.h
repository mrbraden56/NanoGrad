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
    void receive_python_object(int* shape);

private:
    int example;
};

#endif