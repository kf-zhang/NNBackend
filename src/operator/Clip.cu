#include<cassert>

#include"../common/param.h"
#include"./Clip.h"


template<typename T>
__global__ void clip(const T* input,T* output,T min,T max,int size)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<size){
        T tmp = input[idx]<max?input[idx]:max;
        tmp = tmp>min?tmp:min;
        output[idx] = tmp;
    }
}


template<class T> 
Clip<T>::Clip(T min_, T max_):min(min_),max(max_)
{
    
}

template<typename T>
void Clip<T>::operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const
{
    assert( in.size()==1    );
    assert( out.size()==1   );
    Tensor<T>* input = in.at(0);
    Tensor<T>* output = out.at(0);
    
    assert( input->getShape() == output->getShape() );

    int size = input->size();
    
    clip<T><<<ceill((double)size/BLOCK_SIZE),BLOCK_SIZE>>>(input->gpu_pointer(),output->raw_pointer(),min,max,size);
}


template<typename T>
std::vector<std::vector<int>> Clip<T>::outShape(const std::vector<Tensor<T>*> &in ) const
{
    std::vector<std::vector<int>> inShape;
    for(const auto&p:in)
        inShape.push_back(p->getShape());
    return inShape;
}


template class Clip<int>;
template class Clip<float>;
template class Clip<double>;