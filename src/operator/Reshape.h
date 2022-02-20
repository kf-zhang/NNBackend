#pragma once
#include"./Operator.h"

template<typename T>
class Reshape:public Operator<T,T>
{
public:
    void operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const override;
    std::vector<std::vector<int>>outShape(const std::vector<Tensor<T>*> &in) const override;
protected:
    std::vector<int> newShape(int size,std::vector<int> setShape) const;
};

