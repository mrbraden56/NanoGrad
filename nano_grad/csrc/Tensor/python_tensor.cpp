#include "python_tensor.h"

Tensor::Tensor(){
    // constructor implementation
}

double* Tensor::dot(double* x_array, int* x_shape, double* y_array, int* y_shape){
    int M = x_shape[0];
    int L = x_shape[1];//Same as y_shape[0]
    int N = y_shape[1];
    double* z= new double[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            z[j + i * N] = 0;
            for (int k = 0; k < L; k++) {
                z[j + i * N] += x_array[k + i * L] * y_array[j + k * N];
            }
        }
    }

    return z;
}