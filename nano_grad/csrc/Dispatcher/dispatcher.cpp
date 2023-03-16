#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

void Dispatcher::receive_python_object(PyObject* obj){
    if (PyLong_Check(obj)) {
        long value = PyLong_AsLong(obj);
        if (value == -1 && PyErr_Occurred()) {
            PyErr_Print();
            return;
        }
        std::cout << "Got Python Int: " << value << std::endl;
    } else if (PyUnicode_Check(obj)) {
        const char* str = PyUnicode_AsUTF8(obj);
        std::cout << "Got Python String: " << str << std::endl;
    } else {
        std::cout << "Unsupported Python object type" << std::endl;
    }
}

extern "C" void receive_python_object_wrapper(PyObject* obj) {
    static Dispatcher dispatcher;
    dispatcher.receive_python_object(obj);
}
