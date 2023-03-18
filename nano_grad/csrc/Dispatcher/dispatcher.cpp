#include "dispatcher.h"

Dispatcher::Dispatcher(){
    // constructor implementation
}

void Dispatcher::receive_python_object(int* shape){
    // if (PyLong_Check(obj)) {
    //     long value = PyLong_AsLong(obj);
    //     if (value == -1 && PyErr_Occurred()) {
    //         PyErr_Print();
    //         return;
    //     }
    //     std::cout << "Got Python Int: " << value << std::endl;
    // } else if (PyUnicode_Check(obj)) {
    //     const char* str = PyUnicode_AsUTF8(obj);
    //     std::cout << "Got Python String: " << str << std::endl;
    // } else {
    //     std::cout << "Unsupported Python object type" << std::endl;
    // }
    std::vector<int> shape_vec;

    // Copy the shape values from the input array to the vector
    for (int i = 0; i < 2; i++) {
        shape_vec.push_back(shape[i]);
    }

    // Print the shape values
    std::cout << "Shape: ";
    for (int i = 0; i < shape_vec.size(); i++) {
        std::cout << shape_vec[i] << " ";
    }
    std::cout<<"\n";
}

extern "C" void receive_python_object_wrapper(int* shape) {
    static Dispatcher dispatcher;
    dispatcher.receive_python_object(shape);
}
