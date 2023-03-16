#include <Python.h>
#include <string.h>
#include <iostream>


#ifndef DISPATCHER_H
#define DISPATCHER_H

class Dispatcher {
public:
    Dispatcher(); // constructor
    void printMessage(); // member function
    void receive_python_object(PyObject* obj);

private:
    int example;
};

#endif