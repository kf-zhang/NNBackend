#include<cassert>

#include"../common/param.h"
#include"./Clip.h"


template<typename T>
__global__ void clip(const T* input,T* output,const T* min,const T* max,int size)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<size){
        T tmp = input[idx]<*max?input[idx]:*max;
        tmp = tmp>*min?tmp:*min;
        output[idx] = tmp;
    }
}


template<typename T>
void Clip<T>::operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const
{
    assert( in.size()==3    );
    assert( out.size()==1   );
    Tensor<T>* input = in.at(0);
    Tensor<T>* min = in.at(1);
    Tensor<T>* max = in.at(2);
    Tensor<T>* output = out.at(0);
    
    assert( input->getShape() == output->getShape() );
    assert( min->size()==1 && max->size()==1 );
    int size = input->size();
    
    clip<T><<<ceill((double)size/BLOCK_SIZE),BLOCK_SIZE>>>(input->gpu_pointer(),output->raw_pointer(),min->gpu_pointer(),max->gpu_pointer(),size);
}



template class Clip<int>;
template class Clip<float>;
template class Clip<double>;