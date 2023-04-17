#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

double* Dispatcher::receive_dot_product(double* x_array, int* x_shape, double* y_array, int* y_shape, int* python_object){
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
    double* output;

    if(this->instances.find(instance)==this->instances.end()){
        instances.insert(std::make_pair(instance, Ops()));
        output = instances[instance].dot(x_array, x_shape, y_array, y_shape);
    }
    else{
        output = instances[instance].dot(x_array, x_shape, y_array, y_shape);
    }
    return output;
}

extern "C" double* call_receive_dot_product(double* x_array, int* x_shape, double* y_array, int* y_shape, int* python_object) {
    static Dispatcher dispatcher;
    return dispatcher.receive_dot_product(x_array, x_shape, y_array, y_shape, python_object);
}
