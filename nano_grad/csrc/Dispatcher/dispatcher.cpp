#include "dispatcher.h"

static Dispatcher dispatcher;

Dispatcher::Dispatcher(){
    // constructor implementation
}

std::vector<Tensor*> Dispatcher::wrap(double* x, int* x_shape){
    int n = x_shape[0];  // number of rows
    int m = x_shape[1];  // number of columns
    std::vector<Tensor*> tensors(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::function<void()> _backward_empty = std::function<void()>();
            std::vector<Tensor> _prev_empty = std::vector<Tensor>();
            std::function<void()>& _backward = _backward_empty;
            std::vector<Tensor>& _prev = _prev_empty;
            double grad=0;
            Tensor *tensor = new Tensor(&x[i * m + j], grad, _backward, _prev);
            tensors[i * m + j]=tensor;
        }
    }

    return tensors;
}

std::vector<Tensor*> Dispatcher::receive_dot_product(double* x_array, int* x_shape, double* y_array, int* y_shape, int* python_object){
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
    std::vector<Tensor*> output;

    std::vector<Tensor*> x_array_tensor;
    std::vector<Tensor*> y_array_tensor = this->wrap(y_array, y_shape);
    //No instance, firstmake forward pass
    if(this->instances.find(instance)==this->instances.end()){
        x_array_tensor = this->wrap(x_array, x_shape);
        instances.insert(std::make_pair(instance, Ops()));
        output = instances[instance].dot(x_array_tensor, x_shape, y_array_tensor, y_shape);
        this->_prev_call = output;
    }
    //instance exists, nth forward pass
    else{
        x_array_tensor = this->_prev_call;
        instances.insert(std::make_pair(instance, Ops()));
        output = instances[instance].dot(x_array_tensor, x_shape, y_array_tensor, y_shape);//here
        this->_prev_call = output;
    }
    return output;
}

extern "C" double* call_receive_dot_product(double* x_array, int* x_shape, double* y_array, int* y_shape, int* python_object) {
    std::vector<Tensor*> result = dispatcher.receive_dot_product(x_array, x_shape, y_array, y_shape, python_object);
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
    dispatcher_instances.insert(std::make_pair(instance, dispatcher));

    int M = x_shape[0];
    int N = y_shape[1];
    int size = M * N;
    
    double* output_data = new double[size];
    for (int i = 0; i < size; i++) {
        output_data[i] = *result[i]->data;
    }
    
    return output_data;
}

//TODO: Figure out how we should structure 'parameters' for interaction between Python and C++
extern "C" void parameters(int* python_object) {
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
    Dispatcher dispatcher = dispatcher_instances[instance];
    // std::vector<Tensor*> result = dispatcher.receive_dot_product(x_array, x_shape, y_array, y_shape, python_object);
    

    
    // return output_data;
}
