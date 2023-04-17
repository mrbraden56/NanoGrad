#include "tensor.h"

Tensor::Tensor(double* data, double grad, std::function<void()> _backward, std::vector<Tensor> _prev){
    this->data=data;
    this->grad=grad;
    this->_backward=_backward;
    this->_prev=_prev;
}

Tensor Tensor::operator+(Tensor& other){
    double* new_data = new double;
    *new_data=*(this->data) + (*other.data);
    int grad=0;
    std::vector<Tensor> _prev{ *this, other };
    Tensor out=Tensor(new_data, grad, nullptr, _prev);

    std::function<void()> _backward = [&]() mutable{
        this->grad =  this->grad + out.grad;
        other.grad = other.grad + out.grad;

    };
    out._backward = _backward;
    return out;
}

Tensor Tensor::operator*(Tensor& other){
    double* new_data = new double;
    *new_data=*(this->data) * (*other.data);
    int grad=0;
    std::vector<Tensor> _prev{ *this, other };
    Tensor out=Tensor(new_data, grad, nullptr, _prev);

    std::function<void()> _backward = [&]() mutable{
        this->grad =  this->grad + (*(other.data) * out.grad);
        other.grad = other.grad + (*(this->data) * out.grad);
    };
    out._backward = _backward;
    return out;
}