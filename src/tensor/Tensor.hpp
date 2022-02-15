#pragma once
#include<iostream>
#include <memory>
#include<vector>
#include<cuda_runtime.h>

#include"../common/util.hpp"

template<typename T>
class Tensor
{
protected:
    std::vector<int> shape;
    T* data;
public:
    Tensor( const std::vector<int> &s = {}, T *d = nullptr);
    Tensor(const Tensor<T> &s);
    Tensor &operator=(const Tensor &other);
    int size() const;
    int dim()  const;
    std::vector<int> getShape() const;
    const T* gpu_pointer() const;//return exactly the data
    T*  raw_pointer();
    std::unique_ptr<T> cpu_pointer() const;//alllocate a memory in cpu, copy data from gpu to cpu
    virtual ~Tensor();
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& t);

//d应该是指向cpu内存的指针
template<typename T>
Tensor<T>::Tensor(const std::vector<int> &s , T *d)
:shape(s)
{
    int dim = shape.size();
    if(dim)
    {
        gpuErrchk( cudaMalloc(&data, size() * sizeof(T)) );
    }
    else
    {
        this->data = nullptr;
    }   
    if(d)
        gpuErrchk(cudaMemcpy(data, d, size() * sizeof(T), cudaMemcpyHostToDevice));
}

//拷贝构造函数
template<typename T>
Tensor<T>::Tensor(const Tensor<T> &s)
:shape(s.shape)
{
    if(data)
        gpuErrchk(cudaFree(data));
    
    if (size())
    {
        gpuErrchk(cudaMalloc(&data, size() * sizeof(T)));
        gpuErrchk(cudaMemcpy(data, s.pointer(), size() * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    else
        data = nullptr;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor &other)
{
    if( this == &other )
        return *this;
    
    if(data)
        gpuErrchk(cudaFree(data));
    shape = other.getShape();

    gpuErrchk(cudaMalloc(&data, size() * sizeof(T)));
    gpuErrchk(cudaMemcpy(data, other.gpu_pointer(), size() * sizeof(T), cudaMemcpyDeviceToDevice));
    
    return *this;
}

//返回tensor中数据个数
template<typename T>
int Tensor<T>::size() const
{
    if (shape.size() == 0)
            return 0;

    int size = 1;
    for (int i : this->shape)
    {
        size *= i;
    }
    return size;
}

//返回tensor中数据的维度
template<typename T>
int Tensor<T>::dim()  const
{
    return shape.size();
}

//返回tensor中数据的形状
template<typename T>
std::vector<int> Tensor<T>::getShape() const
{
    return shape;
}

//直接返回指针 data(!指针指向gpu)
template<typename T>
const T* Tensor<T>::gpu_pointer() const
{
    return (const T*)(data);
}

//返回指针data
template<typename T>
T* Tensor<T>::raw_pointer()
{
    return data;
}

//在cpu中申请一块内存,将gpu中的数据复制到cpu中,返回指向cpu内存的指针
template<typename T>
std::unique_ptr<T> Tensor<T>::cpu_pointer() const
{   
    int len = size()*sizeof(T);
    if(len){
        T* p = new T[len];
        gpuErrchk( cudaMemcpy(p,data,len,cudaMemcpyDeviceToHost) );
        return std::unique_ptr<T>(p);
    }
    else
        return std::unique_ptr<T>(nullptr);
}

//释放data的内存
template<typename T>
Tensor<T>::~Tensor()
{
    if(data)
        gpuErrchk( cudaFree(data) );
}


//打印tensor的形状以及数据，由于涉及到数据的复制，会比较慢
template<typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& t){
    std::vector<int> shape = t.getShape();
    int size = t.size();
    out<<"Tensor shape:(";
    for(int i:shape)
        out<<i<<",";
    out<<")\n";
    std::unique_ptr<T> p = t.cpu_pointer();
    for(int i=0;i<size;i++)
        out<<*(p.get()+i)<<' ';
    return out;
}


