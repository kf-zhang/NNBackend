#include<cassert>

#include"BatchNormalization.h"

// Y = (X - mean) / sqrt(var + epsilon) * scale + B
//grid[N,C] block[K]
//X:[N,C,K] Y[N,C,K] 
//meax[C] scale[C] B[C] var[C]
template<typename T>
__global__ void transform(const T* X,T*Y,const T* mean,const T* var,const T* scale,const T* B,float epsilon)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int k = threadIdx.x;

    int K = blockDim.x;
    int C = gridDim.y;
    // printf("kernel K:%d C:%d\n",K,C);
    int idx = n*K*C + c * K + k;
    Y[idx] = ((X[idx] - mean[c] )/sqrt(epsilon + var[c])) *scale[c] + B[c];
    // printf("kernel n:%d c:%d k:%d idx:%d X[idx]:%f Y[idx]:%f\n",n,c,k,idx,X[idx],Y[idx]);
}


template<typename T> 
BatchNormalization<T>::BatchNormalization(float eps, float momen, int train_mode ):
epsilon(eps),
momentum(momen),
training_mdoe(train_mode)
{

}

//调用算子进行前向传播
template<typename T> 
void BatchNormalization<T>::operator()(const std::vector<Tensor<T> *> &in, const std::vector<Tensor<T> *> &out) const
{
    if(training_mdoe)
    {
        std::cerr<<"unimplementated training mode in BatchNormalization"<<std::endl;
        return;
    }
    else
    {
        assert( in.size() == 5);
        Tensor<T>* X = in.at(0);
        Tensor<T>* scale = in.at(1);
        Tensor<T>* B = in.at(2);
        Tensor<T>* mean = in.at(3);
        Tensor<T>* var = in.at(4);

        assert( out.size() == 1);
        Tensor<T>* Y = out.at(0);

        std::vector<int> X_shape = X->getShape();
        int N = X_shape.at(0);
        int C = X_shape.at(1);
        int K = X->size()/N/C;

        dim3 grid(N,C);
        dim3 blk(K);
        std::cout<<"N: "<<N<<" C: "<<C<<" K: "<<K<<std::endl;
        transform<T><<<grid,blk>>>  ( 
                                        X->gpu_pointer(),Y->raw_pointer(),
                                        mean->gpu_pointer(),var->gpu_pointer(),
                                        scale->gpu_pointer(),B->gpu_pointer(),
                                        epsilon
                                    );
    }
}


template<typename T>
std::vector<std::vector<int>> BatchNormalization<T>::outShape(const std::vector<std::vector<int>> &inShape) const
{
    assert(inShape.size()>=1);
    std::vector<int> v(inShape.at(0));

    return std::vector<std::vector<int>>({v});
}


template class BatchNormalization<int>;
template class BatchNormalization<float>;
template class BatchNormalization<double>;