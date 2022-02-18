#pragma once
#include"../tensor/Tensor.hpp"

template<typename Tin,typename Tout = Tin>
class Operator
{
private:
    /* data */
public:
    virtual void operator()(const std::vector<Tensor<Tin>*> &in,const std::vector<Tensor<Tout>*> &out ) const =0;
    virtual std::vector<std::vector<int>>outShape(const std::vector< std::vector<int> >&inShape);
};


