#pragma once
#include"./Operator.h"

template<typename T>
class Clip:public Operator<T,T>
{
public:
    Clip(T min_,T max_);
    void operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const override;
    std::vector<std::vector<int>>outShape(const std::vector< std::vector<int> >&inShape) const override;
private:
    T min;
    T max;
};