#include"GlobalAveragePool.h"
#include"../common/param.h"
#include<cassert>

//X[M,N]-> Y[M]
//对X最后的维度进行reduce(平均),得到Y
template<typename T>
__global__ void avgpool(const T*X,T*Y,int M,int N)
{
    extern __shared__ char smem[];

    T* sharedData = (T*) smem;

    int m = blockIdx.x;
    int n = threadIdx.x;
    sharedData[n] = 0;

    if(n<N)
        sharedData[n] = X[m*N+n];
    __syncthreads();

    for(int stride = blockDim.x/2;stride>0;stride>>=1 )
    {
        if(n<stride)
            sharedData[n]+=sharedData[n+stride];
        __syncthreads();
    }
    if(n==0)
        Y[m] = sharedData[0]/N;
}

//
template<typename T> 
void GlobalAveragePool<T>::operator()(const std::vector<Tensor<T> *> &in, const std::vector<Tensor<T> *> &out) const
{
    assert( in.size() == 1 );
    assert( out.size() == 1 );

    Tensor<T>* X = in.at(0);
    Tensor<T>* Y = out.at(0);

    std::vector<int> X_shape = X->getShape();
    assert(X_shape.size()>=2);

    int N = X_shape.at(X_shape.size()-1) * X_shape.at(X_shape.size()-2);
    int M = X->size()/N;
    int blocksize = WARP_SIZE;
    while(blocksize<N)
        blocksize*=2;
    
    avgpool<T><<<M,blocksize,blocksize*sizeof(T)>>>(X->gpu_pointer(),Y->raw_pointer(),M,N);
}


template<typename T> 
std::vector<std::vector<int>> GlobalAveragePool<T>::outShape(const std::vector<std::vector<int>> &inShape) const
{
    assert( inShape.size() == 1);

    std::vector<int> v(inShape.at(0));
    assert(v.size()>=2);
    v.at(v.size()-1) = 1;
    v.at(v.size()-2) = 1;

    return std::vector<std::vector<int>>({v});
}



template class GlobalAveragePool<int>;
template class GlobalAveragePool<float>;
template class GlobalAveragePool<double>;