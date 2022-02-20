#include<cassert>

#include"../common/param.h"
#include"./Add.h"
//计算element wise加法 C = A + B
template<typename T>
__global__ void add(const T*A,const T*B,T*C,unsigned int size)
{
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<size)
        C[idx] = A[idx] + B[idx];
}

//Add 的前向传播函数,必须满足 in={A,B} out = {C},A,B,C的shape必须相同.
//令C=A+B,暂时未实现broadcast
template<typename T>
void Add<T>::operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const
{
    std::cout<<"0\n";
    assert( in.size()==2    );
    assert( out.size()==1   );
    Tensor<T>* A = in.at(0);
    Tensor<T>* B = in.at(1);
    Tensor<T>* C = out.at(0);
    assert( (A->getShape()==B->getShape())&&(A->getShape()==C->getShape()) );
    int size = A->size();
    std::cout<<"1\n";
    add<T><<<ceill((double)size/BLOCK_SIZE),BLOCK_SIZE>>>(A->gpu_pointer(),B->gpu_pointer(),C->raw_pointer(),size);
    std::cout<<"2\n";
}

template<typename T>
std::vector<std::vector<int>> Add<T>::outShape(const std::vector< std::vector<int> >&inShape) const
{
    std::vector<int>  v(inShape.at(0));
    return std::vector<std::vector<int>>({v});
}


template class Add<int>;
template class Add<float>;
template class Add<double>;
