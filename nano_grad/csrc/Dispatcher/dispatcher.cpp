#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

void Dispatcher::receive_python_object(PyObject* obj){
    long value = PyLong_AsLong(obj);
    std::cout << "Got Python Val: " << value << std::endl;
}

extern "C" void receive_python_object_wrapper(PyObject* obj) {
    static Dispatcher dispatcher;
    dispatcher.receive_python_object(obj);
}
