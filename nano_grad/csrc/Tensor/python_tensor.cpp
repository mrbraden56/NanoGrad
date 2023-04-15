#include "python_tensor.h"

Tensor::Tensor(){
    // constructor implementation
}

double* Tensor::dot(double* x_array, int* x_shape, double* y_array, int* y_shape){
    std::cout<<"In python_tensor class!\n";
    int x_rows=x_shape[0];
    int x_cols=x_shape[1];
    for (int i = 0; i < x_rows; i++) {
        for (int j = 0; j < x_cols; j++) {
            int CM = j * x_cols + i;
            int RM = i * x_cols + j;
            std::cout << x_array[RM] << " ";
        }
        std::cout<<"\n";
    }

    int z_rows=x_shape[0];
    int z_cols=y_shape[1];
    double* z= new double[z_rows*z_cols];
    for (int i = 0; i < z_rows; i++) {
        for (int j = 0; j < z_cols; j++) {
            int CM = j * z_cols + i;
            int RM = i * z_cols + j;
            z[RM]=0;
            std::cout << z[RM] << " at location "<<RM;
        }
        std::cout<<"\n";
    }


    // z[0]=(x[0]*y[0]) + (x[0 * x_cols + 0]*y[1])
    // z[1]=(x[1]*y[0]) + (x[0 * x_cols + 1]*y[1])
    // z[2]=(x[2]*y[0]) + (x[0 * x_cols + 2]*y[1])

    // z[3]=(x[0]*y[2]) + (x[0 * x_cols + 0]*y[3])
    // ...
    // z[6]=(x[0]*y[4]) + (x[0 * x_cols + 0]*y[5])
    std::cout<<"Returning...\n";
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