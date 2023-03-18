#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

void Dispatcher::receive_dot_product_shapes(int* x_shape, int x_shape_length, int* y_shape, int y_shape_length){
    std::vector<int> x_shape_vec;
    for (int i = 0; i < x_shape_length; i++) {
        x_shape_vec.push_back(x_shape[i]);
    }

    std::vector<int> y_shape_vec;
    for(int i=0; i<y_shape_length; i++){
        y_shape_vec.push_back(y_shape[i]);
    }

    // Print the shape values
    std::cout << "Shape: ";
    for (int i = 0; i < x_shape_vec.size(); i++) {
        std::cout << x_shape_vec[i] << " ";
    }
    std::cout<<"\n";

    std::cout << "Shape: ";
    for (int i = 0; i < y_shape_vec.size(); i++) {
        std::cout << y_shape_vec[i] << " ";
    }
    std::cout<<"\n";
}

extern "C" void call_receive_dot_product_shapes(int* x_shape, int x_shape_length, int* y_shape, int y_shape_length) {
    static Dispatcher dispatcher;
    dispatcher.receive_dot_product_shapes(x_shape, x_shape_length, y_shape, y_shape_length);
}
