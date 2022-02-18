#pragma once
#include"./Operator.h"

template<typename T>
class Clip:public Operator<T,T>
{
public:
    void operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const override;
};