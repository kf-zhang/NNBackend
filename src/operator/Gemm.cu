#include<cassert>

#include"Gemm.h"
#include"../common/param.h"
//计算GEMM,没有C
//kerneltye:0
template<typename T>
__global__ void kernel0(const T*A,const T*B,T*Y,int M,int K,int N,float alpha,int transA,int transB)
{
    T c = 0;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    //A[M,K] B[K,N] C[M,N] Y[M,N]
    //Y[x,y] = \sum_i A[x,i]*B[i,y] * alpha + beta * C[x,y]
    if(!transA && ! transB)
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*K+i]*B[i*N+y];
            }
            Y[x*N+y] = alpha * c;
        }
    }
    //A[M,K] B[N,K] C[M,N] Y[M,N]
    //Y[x,y] = \sum_i A[x,i]*B[y,i] * alpha + beta * C[x,y]
    else if( !transA && transB )
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*K+i]*B[y*K+i];
            }
            Y[x*N+y] = alpha * c ;
        }
    }
    //A[K,M] B[K,N] C[M,N] Y[M,N]
    //Y[x,y] = \sum_i A[i,x]*B[i,y] * alpha + beta * C[x,y]
    else if( transA && !transB )
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[i*M+x]*B[i*N+y];
            }
            Y[x*N+y] = alpha * c ;
        }
    }
    //A[K,M] B[N,K] C[M,N] Y[M,N]
    //Y[x,y] = \sum_i A[i,x]*B[y,i] * alpha + beta * C[x,y]
    else if(transA && transB)
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*M+x]*B[y*K+i];
            }
            Y[x*N+y] = alpha * c ;
        }
    }
}



//计算GEMM,C的形状为(M,N)
//kerneltype 1
template<typename T>
__global__ void kernel1(const T*A,const T*B,const T*C,T*Y,int M,int K,int N,float alpha,float beta,int transA,int transB)
{
    T c = 0;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    //A[M,K] B[K,N] C[M,N] Y[M,N]
    //Y[x,y] = \sum_i A[x,i]*B[i,y] * alpha + beta * C[x,y]
    if(!transA && ! transB)
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*K+i]*B[i*N+y];
            }
            Y[x*N+y] = alpha * c + beta * C[x*N+y];
        }
    }
    //A[M,K] B[N,K] C[M,N] Y[M,N]
    //Y[x,y] = \sum_i A[x,i]*B[y,i] * alpha + beta * C[x,y]
    else if( !transA && transB )
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*K+i]*B[y*K+i];
            }
            Y[x*N+y] = alpha * c + beta * C[x*N+y];
        }
    }
    //A[K,M] B[K,N] C[M,N] Y[M,N]
    //Y[x,y] = \sum_i A[i,x]*B[i,y] * alpha + beta * C[x,y]
    else if( transA && !transB )
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[i*M+x]*B[i*N+y];
            }
            Y[x*N+y] = alpha * c + beta * C[x*N+y];
        }
    }
    //A[K,M] B[N,K] C[M,N] Y[M,N]
    //Y[x,y] = \sum_i A[i,x]*B[y,i] * alpha + beta * C[x,y]
    else if(transA && transB)
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*M+x]*B[y*K+i];
            }
            Y[x*N+y] = alpha * c + beta * C[x*N+y];
        }
    }
}


//计算GEMM,C的形状为(M,1)
//kerneltype 2
template<typename T>
__global__ void kernel2(const T*A,const T*B,const T*C,T*Y,int M,int K,int N,float alpha,float beta,int transA,int transB)
{
    T c = 0;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    //A[M,K] B[K,N] C[M,1] Y[M,N]
    //Y[x,y] = \sum_i A[x,i]*B[i,y] * alpha + beta * C[x]
    if(!transA && ! transB)
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*K+i]*B[i*N+y];
            }
            Y[x*N+y] = alpha * c + beta * C[x];
        }
    }
    //A[M,K] B[N,K] C[M,1] Y[M,N]
    //Y[x,y] = \sum_i A[x,i]*B[y,i] * alpha + beta * C[x]
    else if( !transA && transB )
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*K+i]*B[y*K+i];
            }
            Y[x*N+y] = alpha * c + beta * C[x];
        }
    }
    //A[K,M] B[K,N] C[M,1] Y[M,N]
    //Y[x,y] = \sum_i A[i,x]*B[i,y] * alpha + beta * C[x]
    else if( transA && !transB )
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[i*M+x]*B[i*N+y];
            }
            Y[x*N+y] = alpha * c + beta * C[x];
        }
    }
    //A[K,M] B[N,K] C[M,1] Y[M,N]
    //Y[x,y] = \sum_i A[i,x]*B[y,i] * alpha + beta * C[x]
    else if(transA && transB)
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*M+x]*B[y*K+i];
            }
            Y[x*N+y] = alpha * c + beta * C[x];
        }
    }
}


//计算GEMM,C的形状为(1,N)
//kerneltype 3
template<typename T>
__global__ void kernel3(const T*A,const T*B,const T*C,T*Y,int M,int K,int N,float alpha,float beta,int transA,int transB)
{
    T c = 0;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    //A[M,K] B[K,N] C[1,N] Y[M,N]
    //Y[x,y] = \sum_i A[x,i]*B[i,y] * alpha + beta * C[y]
    if(!transA && ! transB)
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*K+i]*B[i*N+y];
            }
            Y[x*N+y] = alpha * c + beta * C[y];
        }
    }
    //A[M,K] B[N,K] C[1,N] Y[M,N]
    //Y[x,y] = \sum_i A[x,i]*B[y,i] * alpha + beta * C[y]
    else if( !transA && transB )
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*K+i]*B[y*K+i];
            }
            Y[x*N+y] = alpha * c + beta * C[y];
        }
    }
    //A[K,M] B[K,N] C[1,N] Y[M,N]
    //Y[x,y] = \sum_i A[i,x]*B[i,y] * alpha + beta * C[y]
    else if( transA && !transB )
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[i*M+x]*B[i*N+y];
            }
            Y[x*N+y] = alpha * c + beta * C[y];
        }
    }
    //A[K,M] B[N,K] C[1,N] Y[M,N]
    //Y[x,y] = \sum_i A[i,x]*B[y,i] * alpha + beta * C[y]
    else if(transA && transB)
    {
        if(x<M&&y<N)
        {
            for(int i=0;i<K;i++)
            {
                c+=A[x*M+x]*B[y*K+i];
            }
            Y[x*N+y] = alpha * c + beta * C[y];
        }
    }
}


template<typename T> 
Gemm<T>::Gemm(float al, float be, int tA, int tB):
alpha(al),
beta(be),
transA(tA),
transB(tB)
{

}

//根据不同的输入shape使用不同的核函数
//-1 表示输入shape不合法
template<typename T>
int Gemm<T>::kernelType(const std::vector< std::vector<int> >&inShape) const
{
    int numOfInput = inShape.size();

    if( numOfInput!=2 && numOfInput!=3 )
    {
        std::cerr<<"The number of input is "<<numOfInput<<" which is invalid in Gemm"<<std::endl;
        return -1;
    }
    if(numOfInput==2)
        return 0;

    int M,N,K;
    auto A_shape = inShape.at(0);
    auto B_shape = inShape.at(1);
    auto C_shape = inShape.at(2);
    getParam(A_shape,B_shape,M,N,K);

    if( C_shape.size()==2 )
    {
        if(C_shape.at(0)==M&&C_shape.at(1)==N)
            return 1;
        else if(C_shape.at(0)==M&&C_shape.at(1)==1)
            return 2;
        else if(C_shape.at(0)==1&&C_shape.at(1)==N)
            return 3;
        else
        {
            std::cerr<<"The shape of C is ("<<C_shape.at(0)<<","<<C_shape.at(1)<<"), which is invalid in Gemm";
            return -1;
        }
    }
    else if( C_shape.size() == 1 )
    {
        if(C_shape.at(0)==N)
            return 3;
        else
        {
            std::cerr<<"The shape of C is ("<<C_shape.at(0)<<","<<C_shape.at(1)<<"), which is invalid in Gemm";
            return -1;
        }
    }
    else
    {
        std::cerr<<"The dim of C is "<<C_shape.size()<<" which is invalid in Gemm"<<std::endl;
        return -1;
    }
}


template<typename T> 
void Gemm<T>::operator()(const std::vector<Tensor<T> *> &in, const std::vector<Tensor<T> *> &out) const
{
    assert(in.size()==3||in.size()==2);
    assert(out.size()==1);
    Tensor<T> * A = in.at(0);
    Tensor<T> * B = in.at(1);
    Tensor<T> * Y = out.at(0);
    int M,N,K;
    getParam(A->getShape(),B->getShape(),M,N,K);

    dim3 grid(ceil((double)M/BLOCK_SIZE),ceil((double)N/BLOCK_SIZE));
    dim3 blk(BLOCK_SIZE,BLOCK_SIZE);


    std::vector<std::vector<int>> inShapes;
    for(const auto p:in)
        inShapes.push_back( p->getShape() );
    int kType = kernelType(inShapes);
    switch (kType)
    {
        case 0:
            kernel0<T><<<grid,blk>>>(A->gpu_pointer(),B->gpu_pointer(),Y->raw_pointer(),M,K,N,alpha,transA,transB);
            break;
        case 1:
            {
                Tensor<T> * C = in.at(2);
                kernel1<T><<<grid,blk>>>(A->gpu_pointer(),B->gpu_pointer(),C->gpu_pointer(),Y->raw_pointer(),M,K,N,alpha,beta,transA,transB);
            }
            break;
        case 2:
            {
                Tensor<T> * C = in.at(2);
                kernel2<T><<<grid,blk>>>(A->gpu_pointer(),B->gpu_pointer(),C->gpu_pointer(),Y->raw_pointer(),M,K,N,alpha,beta,transA,transB);
            }
            break;
        case 3:
            {
                Tensor<T> * C = in.at(2);
                kernel3<T><<<grid,blk>>>(A->gpu_pointer(),B->gpu_pointer(),C->gpu_pointer(),Y->raw_pointer(),M,K,N,alpha,beta,transA,transB);
            }
            break;
        default:
            std::cerr<<"invalid kernel type"<<std::endl;
            break;
    }
}

//获取M,N,K
template<typename T>
void Gemm<T>::getParam(const std::vector<int>& A_shape,const std::vector<int>& B_shape,int &M,int&N,int& K)const
{
    int Ka;
    int Kb;

    if(transA)
    {  
        M = A_shape.at(1);
        Ka = A_shape.at(0);
    }
    else
    {
        M = A_shape.at(0);
        Ka = A_shape.at(1);
    }

    if(transB)
    {  
        Kb = B_shape.at(1);
        N = B_shape.at(0);
    }
    else
    {
        Kb = B_shape.at(0);
        N = B_shape.at(1);
    }
    
    assert(Ka==Kb);
    K = Ka;
    
}


//根据输入A,B的形状,计算结果Y的形状并返回
template<typename T> 
std::vector<int> Gemm<T>::output_shape(const std::vector<int> &A_shape, const std::vector<int> &B_shape) const
{
    assert(A_shape.size()==2);
    assert(B_shape.size()==2);
    int M,N,K;

    getParam(A_shape,B_shape,M,N,K);

    std::vector<int> v;
    v.push_back(M);
    v.push_back(N);

    return v;
}


//计算输出tensor的形状
template<typename T>
std::vector<std::vector<int>> Gemm<T>::outShape(const std::vector<Tensor<T>*> &in) const
{
    assert( in.size()>=2 );
    std::vector<std::vector<int>> v;
    v.push_back( output_shape(in.at(0)->getShape(),in.at(1)->getShape()) );
    return v;
}


template class Gemm<int>;
template class Gemm<float>;
template class Gemm<double>;

