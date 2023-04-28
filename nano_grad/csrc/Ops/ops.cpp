#include "ops.h"

Ops::Ops(){
    // constructor implementation
}

std::vector<std::shared_ptr<Tensor>> Ops::dot(std::vector<std::shared_ptr<Tensor>> x_array, int* x_shape, 
                              std::vector<std::shared_ptr<Tensor>> y_array, int* y_shape){
    int M = x_shape[0];
    int L = x_shape[1]; // Same as y_shape[0]
    int N = y_shape[1];
    
    std::vector<std::shared_ptr<Tensor>> z(M * N);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::function<void()> _backward_empty = std::function<void()>();
            std::vector<Tensor> _prev_empty = std::vector<Tensor>();
            std::function<void()>& _backward = _backward_empty;
            std::vector<Tensor>& _prev = _prev_empty;
            std::shared_ptr<double> data = std::make_shared<double>(0);
            double grad = 0;
            std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(data, grad, _backward, _prev);
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
