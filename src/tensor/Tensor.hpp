#pragma once
#include<iostream>
#include <memory>
#include<vector>
#include<cuda_runtime.h>
#include<cassert>

#include"../common/util.hpp"

template<typename T>
class Tensor
{
protected:
    std::vector<int> shape;
    T* data;
public:
    Tensor( const std::vector<int> &s = {}, T *d = nullptr,int fromDevice = 0);
    Tensor(const Tensor<T> &s);
    Tensor &operator=(const Tensor &other);
    int size() const;
    int dim()  const;
    std::vector<int> getShape() const;
    const T* gpu_pointer() const;//return exactly the data
    T*  raw_pointer();
    std::unique_ptr<T> cpu_pointer() const;//alllocate a memory in cpu, copy data from gpu to cpu
    bool setmem(const Tensor<T>& tensor);
    bool reshape(const std::vector<int>& newshape);
    void printShape()const;
    virtual ~Tensor();
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& t);

//构造tensor,s为tensor形状,d为指向数据的指针,fromDevice表明指向device还是host
template<typename T>
Tensor<T>::Tensor(const std::vector<int> &s , T *d,int fromDevice)
:shape(s)
{
    if(size())
    {
        gpuErrchk( cudaMalloc(&data, size() * sizeof(T)) );
    }
    else
    {
        this->data = nullptr;
    }   
    if(d)
    {
        if(fromDevice)
        {
            gpuErrchk( cudaMemcpy(data, d, size() * sizeof(T), cudaMemcpyDeviceToDevice) );
        }
        else
        {
            gpuErrchk( cudaMemcpy(data, d, size() * sizeof(T), cudaMemcpyHostToDevice) );
        }
    }
}

//拷贝构造函数,
template<typename T>
Tensor<T>::Tensor(const Tensor<T> &s)
:shape(s.shape)
{
    if (size())
    {
        gpuErrchk(cudaMalloc(&data, size() * sizeof(T)));
        gpuErrchk(cudaMemcpy(data, s.gpu_pointer(), size() * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    else
        data = nullptr;
}

//赋值函数,会释放掉之前的内存,重新申请内存,然后拷贝数据
template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor &other)
{
    if( this == &other )
        return *this;
    
    if(data)
        gpuErrchk(cudaFree(data));
    shape = other.getShape();

    if( size() )
    {
        gpuErrchk(cudaMalloc(&data, size() * sizeof(T)));
        gpuErrchk(cudaMemcpy(data, other.gpu_pointer(), size() * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    else
    {
        data = nullptr;
    }
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


//复制tensor中的数据,两者必须做到形状相同
template<typename T>
bool Tensor<T>::setmem(const Tensor<T>& tensor)
{
    assert( size()==tensor.size() ); 
    gpuErrchk( cudaMemcpy(data,tensor.gpu_pointer(),size()*sizeof(T),cudaMemcpyDeviceToDevice) );
    return true;
}

//对tensor进行reshape,如果size不同则返回false
template<class T> 
bool Tensor<T>::reshape(const std::vector<int> &newshape)
{
    int num = 1;
    if(!newshape.size())
        num = 1;
    else
        for(int i:newshape)
            num*=i;
    if(num!=size())
        return false;
    shape = newshape;
    return true;
}

template<typename T>
void Tensor<T>::printShape() const
{
    std::cout<<"(";
    for(const auto i:shape)
        std::cout<<i<<",";
    std::cout<<")\n";
}

//打印tensor的形状以及数据，由于涉及到数据的复制，会比较慢
template<typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& t)
{
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





