#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

void Dispatcher::receive_dot_product(double* x_array, int* x_shape, int* y_shape, int y_shape_length, int* python_object){
    PyObject* instance = reinterpret_cast<PyObject*>(*python_object);
    std::cout<<instance<<"\n";
    int rows=x_shape[0];
    int cols=x_shape[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << x_array[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout<<"\n";

    std::vector<int> y_shape_vec;
    for(int i=0; i<y_shape_length; i++){
        y_shape_vec.push_back(y_shape[i]);
    }

    std::cout << "Shape: ";
    for (int i = 0; i < y_shape_vec.size(); i++) {
        std::cout << y_shape_vec[i] << " ";
    }
    std::cout<<"\n";

    //if py_obj exists call that 
    if(this->instances.find(instance)==this->instances.end()){
        std::cout<<"Instance does not exist, creating one\n";
        instances.insert(std::make_pair(instance, Tensor()));
    }
    else{
        std::cout<<"Instance does exist\n";
        instances[instance].dot();
    }
}

extern "C" void call_receive_dot_product(double* x_array, int* x_shape, int* y_shape, int y_shape_length, int* python_object) {
    static Dispatcher dispatcher;
    dispatcher.receive_dot_product(x_array, x_shape, y_shape, y_shape_length, python_object);
}
