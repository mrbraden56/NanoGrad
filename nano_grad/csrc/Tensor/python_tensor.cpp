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
    for (int i = 0; i < z_rows; i++) {
        for (int j = 0; j < z_cols; j++) {
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
    for(int j=0; j<y_cols; j++){
        for(int i=0; i<x_rows; i++){
            z[j * z_cols + i]=(x[i]*y[j]) + ()
        }
    }




    int y_column_counter=0;
    for(int k=0; k<z_rows*z_cols; k++){
        for(int i=0; i<y_cols; i++){
            for (int j = 0; j < x_rows*x_cols; j++) {
                std::unique_ptr<double[]> x_vector(new double[x_rows]);
                if(j%x_rows==0){
                    y_column_counter++;
                }

                z[k]=x[j]*y[y_column_counter] + z[j];
            }
        }
    } 

}

double* Tensor::matrix_vector(double* x, double* y){

}