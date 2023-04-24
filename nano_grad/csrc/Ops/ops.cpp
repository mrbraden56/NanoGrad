#include "ops.h"

Ops::Ops(){
    // constructor implementation
}

std::vector<Tensor*> Ops::dot(std::vector<Tensor*> x_array, int* x_shape, 
                              std::vector<Tensor*> y_array, int* y_shape){
    int M = x_shape[0];
    int L = x_shape[1]; // Same as y_shape[0]
    int N = y_shape[1];
    
    std::vector<Tensor*> z(M * N);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::function<void()> _backward_empty = std::function<void()>();
            std::vector<Tensor> _prev_empty = std::vector<Tensor>();
            std::function<void()>& _backward = _backward_empty;
            std::vector<Tensor>& _prev = _prev_empty;
            double* data = new double(0);
            double grad = 0;
            Tensor *tensor = new Tensor(data, grad, _backward, _prev);
            z[j + i * N] = tensor;
            for (int k = 0; k < L; k++) {
                // *z[j + i * N] = *z[j + i * N] + (*x_array[k + i * L] * *y_array[j + k * N]);
                *z[j + i * N] = *x_array[k + i * L] * *y_array[j + k * N];
                z[j + i * N]->test_parents(*x_array[k + i * L]->data, *y_array[j + k * N]->data);
                // std::cout<<z[j + i * N]->depth()<<"\n";

                // for (int idx = 0; idx < z[j + i * N]->_prev.size(); idx++) {
                //     std::cout << "z[" << j + i * N << "]->_prev[" << idx << "]: " << *z[j + i * N]->_prev[idx].data << "\n";
                // }

            }
        }
    }

    return z;
}
