#pragma once
#include"./Operator.h"

template<typename T>
class BatchNormalization: public Operator<T,T>
{
public:
    BatchNormalization(float eps = 1e-5,float momen=0.9,int train_mode=0);
    void operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const override;
    std::vector<std::vector<int>>outShape(const std::vector< std::vector<int> >&inShape) const override;
private:
    float epsilon;
    float momentum;
    int training_mdoe;
};