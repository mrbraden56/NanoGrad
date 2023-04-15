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
            int index=j * x_cols + i;
            // std::cout << x_array[index] << " ";
            x[index] = x_array[index];
        }
        // std::cout << std::endl;
    }

    int y_rows=y_shape[0];
    int y_cols=y_shape[1];
    double* y = new double[y_rows*y_cols];
    for (int i = 0; i < y_rows; i++) {
        for (int j = 0; j < y_cols; j++) {
            int y_index=j * y_cols + i;
            y[y_index] = y_array[y_index];
        }
    }

    int z_rows=x_shape[0];
    int z_cols=y_shape[1];
    double* z= new double[z_rows*z_cols];
    for (int i = 0; i < x_rows; i++) {
        for (int j = 0; j < y_cols; j++) {
            int z_index=j * z_cols + i;
            z[z_index] = 0;
        }
    }


    // z[0]=(x[0]*y[0]) + (x[0 * x_cols + 0]*y[1])
    // z[1]=(x[1]*y[0]) + (x[0 * x_cols + 1]*y[1])
    // z[2]=(x[2]*y[0]) + (x[0 * x_cols + 2]*y[1])

    // z[3]=(x[0]*y[2]) + (x[0 * x_cols + 0]*y[3])
    // ...
    // z[6]=(x[0]*y[4]) + (x[0 * x_cols + 0]*y[5])

    int lead_dim_A=x_shape[0];
    int lead_dim_B=y_shape[0];
    int lead_dim_C=z_rows;
    for (int i = 0; i < x_rows; i++) {
        for (int j = 0; j < y_cols; j++) {
            std::cout<<z[j * z_cols + i]<<"\n";
        }
    }
    MyGemm(x_shape[0], y_shape[1], x_shape[1], x, lead_dim_A, y, lead_dim_B, z, lead_dim_C);
    for (int i = 0; i < x_rows; i++) {
        for (int j = 0; j < y_cols; j++) {
            std::cout<<z[j * z_cols + i]<<"\n";
        }
    }
    std::cout<<"Size: "<<sizeof(z) / sizeof(z[0])<<"\n";
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