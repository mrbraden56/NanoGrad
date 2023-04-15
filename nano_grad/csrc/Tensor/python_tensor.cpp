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

void Tensor::MyGemm(int m, int n, int k, double *A, int ldA, double *B, int ldB, double *C, int ldC ){
for ( int j=0; j<n; j++ ){
    MyGemv( m, k, A, ldA, &B[ (0)*ldB + j ], 1, &C[ (0)*ldC + j ], 1 );
}
}

void Tensor::MyGemv(int m, int n, double *A, int ldA, double *x, int incx, double *y, int incy){
    for ( int j=0; j<n; j++ ){
        Axpy( m, x[ (j)*incx ] , &A[ (0)*ldA + j ], 1, y, incy );
    }
}

void Tensor::Axpy(int n, double alpha, double *x, int incx, double *y, int incy){
    for ( int i=0; i<n; i++ ){
        y[ (i)*incy ]  += alpha * x[ (i)*incx ];   // Fused Multiply-Add
    }
}