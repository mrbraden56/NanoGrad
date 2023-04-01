#include "python_tensor.h"

Tensor::Tensor(){
    // constructor implementation
}

double* Tensor::dot(double* x_array, int* x_shape, double* y_array, int* y_shape){
    std::cout<<"In python_tensor class!\n";
    int x_rows=x_shape[0];
    int x_cols=x_shape[1];
    double* x = new double[x_rows*x_cols];
    for (int i = 0; i < x_rows; i++) {
        for (int j = 0; j < x_cols; j++) {
            std::cout << x_array[i * x_cols + j] << " ";
            int index=i + j * x_rows;
            x[index] = *(x_array + index);
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < x_rows*x_cols; i++) {
        std::cout<<x[i]<<"\n";
        // do something with val
    }

    std::cout<<"\n";
}