#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

std::vector<std::shared_ptr<Tensor>> Dispatcher::wrap(std::shared_ptr<double> x, int* x_shape){
    int n = x_shape[0];  // number of rows
    int m = x_shape[1];  // number of columns
    std::vector<std::shared_ptr<Tensor>> tensors(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::function<void()> _backward_empty = std::function<void()>();
            std::vector<Tensor> _prev_empty = std::vector<Tensor>();
            std::function<void()>& _backward = _backward_empty;
            std::vector<Tensor>& _prev = _prev_empty;
            double grad=0;
            std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(std::make_shared<double>(x.get()[i * m + j]), grad, _backward, _prev);
            tensors[i * m + j]=tensor;
        } 
    }

    return tensors;
}

std::vector<std::shared_ptr<Tensor>> Dispatcher::receive_dot_product(std::shared_ptr<double> x_array, int* x_shape, 
                                                     std::shared_ptr<double> y_array, int* y_shape, 
                                                     std::unique_ptr<int>& python_object){
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
    std::vector<std::shared_ptr<Tensor>> output;

    std::vector<std::shared_ptr<Tensor>> x_array_tensor;
    std::vector<std::shared_ptr<Tensor>> y_array_tensor = this->wrap(y_array, y_shape);
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

extern "C" double* call_receive_dot_product(double* x_array, 
                                             int* x_shape, 
                                             double* y_array, 
                                             int* y_shape, 
                                             int* python_object) {
    static Dispatcher dispatcher;
    std::shared_ptr<double> x_array_ptr(x_array);
    std::shared_ptr<double> y_array_ptr(y_array);
    std::unique_ptr<int> python_object_ptr(python_object);
    std::vector<std::shared_ptr<Tensor>> result = dispatcher.receive_dot_product(
        x_array_ptr, x_shape, y_array_ptr, y_shape, python_object_ptr);
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object_ptr);
    // dispatcher_instances.insert(std::make_pair(instance, dispatcher));
    int M = x_shape[0];
    int N = y_shape[1];
    int size = M * N;
    auto output_data = std::make_unique<double[]>(size);
    for (int i = 0; i < size; i++) {
        output_data[i] = *result[i]->data;
    }
    return output_data.release();
}


//TODO: Figure out how we should structure 'parameters' for interaction between Python and C++
// extern "C" void parameters(int* python_object) {
//     PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
//     Dispatcher dispatcher = dispatcher_instances[instance];
    // std::vector<Tensor*> result = dispatcher.receive_dot_product(x_array, x_shape, y_array, y_shape, python_object);
    

    
    // return output_data;
// }
