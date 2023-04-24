#include "tensor.h"

Tensor::Tensor(double* data, double grad, std::function<void()>& _backward, std::vector<Tensor>& _prev){
    this->data=data;
    this->grad=grad;
    this->_backward=_backward;
    this->_prev=_prev;
}

void Tensor::test_parents(double parent1, double parent2){
    bool result = false;

    if (_prev.size() == 2) {
        // Compare the data pointers of the _prev elements with parent1 and parent2
        if (*(_prev[0].data) == parent1 && *(_prev[1].data) == parent2) {
            result = true;
        } else if (*(_prev[0].data) == parent2 && *(_prev[1].data) == parent1) {
            result = true;
        }
    }

    if (!result) {
        throw std::runtime_error("Parents do not match!");
    }
}

int Tensor::depth() const{
    int max_depth = 0;
    
    for (const auto& parent : _prev) {
        int parent_depth = parent.depth();
        if (parent_depth > max_depth) {
            max_depth = parent_depth;
        }
    }

    return max_depth + 1;
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
    std::vector<Tensor> _prev{ *this, other };
    std::function<void()> temp = [&]() mutable{};
    Tensor out=Tensor(new_data, grad, temp, _prev);

    std::function<void()> _backward = [&]() mutable{
        this->grad =  this->grad + (*(other.data) * out.grad);
        const_cast<Tensor&>(other).grad = other.grad + (*(this->data) * out.grad);
    };
    out._backward = _backward;
    return out;
}