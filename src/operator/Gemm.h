#pragma once
#include"./Operator.h"

template<typename T>
class Gemm:public Operator<T,T>
{
public:
    Gemm(float al=1.0,float be=1.0,int tA=0,int tB=0);
    void operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const override;
    std::vector<std::vector<int>>outShape(const std::vector<Tensor<T>*> &in) const override;
    std::vector<int> output_shape(const std::vector<int>& A_shape,const std::vector<int>& B_shape) const;
    void getParam(const std::vector<int>& A_shape,const std::vector<int>& B_shape,int &M,int&N,int& K) const;
protected:    
    int kernelType(const std::vector< std::vector<int> >&inShape) const;
private:
    float alpha;
    float beta;
    int transA;
    int transB; 
};