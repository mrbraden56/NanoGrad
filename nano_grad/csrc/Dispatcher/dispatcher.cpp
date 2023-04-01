#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

void Dispatcher::receive_dot_product(double* x_array, int* x_shape, double* y_array, int* y_shape, int* python_object){
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
    std::cout<<instance<<"\n";

    //if py_obj exists call that 
    if(this->instances.find(instance)==this->instances.end()){
        std::cout<<"Instance does not exist, creating one\n";
        instances.insert(std::make_pair(instance, Tensor()));
        instances[instance].dot(x_array, x_shape, y_array, y_shape);
    }
    else{
        std::cout<<"Instance does exist\n";
        instances[instance].dot(x_array, x_shape, y_array, y_shape);
    }
}

extern "C" void call_receive_dot_product(double* x_array, int* x_shape, double* y_array, int* y_shape, int* python_object) {
    static Dispatcher dispatcher;
    dispatcher.receive_dot_product(x_array, x_shape, y_array, y_shape, python_object);
}
