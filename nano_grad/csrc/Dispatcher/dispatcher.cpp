#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

Tensor* Dispatcher::wrap(double* x, int* x_shape){
    int n = x_shape[0];  // number of rows
    int m = x_shape[1];  // number of columns
    Tensor* tensors[n*m];
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
}

double* Dispatcher::receive_dot_product(double* x_array, int* x_shape, double* y_array, int* y_shape, int* python_object){
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
    double* output;
    Tensor* x_array_tensor = this->wrap(x_array, x_shape);
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
