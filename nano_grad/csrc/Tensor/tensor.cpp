#include "tensor.h"

Tensor::Tensor(double* data, double grad, std::function<void()>& _backward, std::vector<Tensor>& _prev){
    this->data=data;
    this->grad=grad;
    this->_backward=_backward;
    this->_prev=_prev;
}

Tensor Tensor::operator+(const Tensor& other){
    double* new_data = new double;
    *new_data=*(this->data) + (*other.data);
    double grad=0;
    std::vector<Tensor> _prev{ *this, other };
    std::function<void()> temp = [&]() mutable{};
    Tensor out=Tensor(new_data, grad, temp, _prev);

    std::function<void()> _backward = [&]() mutable{
        this->grad =  this->grad + out.grad;
        const_cast<Tensor&>(other).grad = other.grad + out.grad;

    };
    out._backward = _backward;
    return out;
}

Tensor Tensor::operator*(const Tensor& other){
    double* new_data = new double;
    *new_data=*(this->data) * (*other.data);
    int grad=0;
    std::function<void()> temp = [&]() mutable{};
    Tensor out=Tensor(new_data, grad, temp, _prev);

    std::function<void()> _backward = [&]() mutable{
        this->grad =  this->grad + (*(other.data) * out.grad);
        const_cast<Tensor&>(other).grad = other.grad + (*(this->data) * out.grad);
    };
    out._backward = _backward;
    return out;
}