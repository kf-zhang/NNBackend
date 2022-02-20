#include<cassert>

#include"Reshape.h"


template<class T> std::vector<int> Reshape<T>::newShape(int size, std::vector<int> setShape) const
{
    int unknownIdx = -1;
    int s=1;
    for(int i=0;i<setShape.size();i++)
    {
        if( setShape.at(i)>0 )
            s*=setShape.at(i);
        else
        {
            if(unknownIdx==-1)
                unknownIdx = i;
            else
            {
                std::cerr<<"too many unkoown shape"<<std::endl;
                exit(-1);
            }
        }
    }

    if(unknownIdx!=-1)
    {
        if(size%s==0)
        {
            setShape[unknownIdx] = size/s;
        }
        else
        {
            std::cerr<<"the new shape is incompatible with the size\n"<<std::endl;
            exit(-1);
        }
    }
    else
    {
        assert(s==size);
    }

    return setShape;
}

template<typename T> 
void Reshape<T>::operator()(const std::vector<Tensor<T> *> &in, const std::vector<Tensor<T> *> &out) const
{
    assert(in.size()==2);
    assert(out.size()==1);

    Tensor<T>* X = in.at(0);
    Tensor<T>* Y = out.at(0);

    void* tmp = (void*)(in.at(1));
    Tensor<int> *newshape = (Tensor<int>*)(tmp) ;  
    std::vector<int> v; 
    auto data = newshape->cpu_pointer();
    int size = newshape->size();
    for(int i=0;i<size;i++)
        v.push_back(*(data.get()+i));

    auto setShape = newShape(Y->size(),v);

    Y->reshape(setShape);
    Y->setmem(*X);
}


template<typename T> 
std::vector<std::vector<int>> Reshape<T>::outShape(const std::vector<Tensor<T> *> &in) const
{
    assert( in.size() ==2);
    std::vector<int> v;
    
    void *tmp = (void*)( in.at(1) );
    Tensor<int> *p = (Tensor<int>*)(tmp);
    auto data = p->cpu_pointer();
    int dim = p->size();
    for(int i=0;i<dim;i++)
    {
        int num = *(data.get()+i);
        v.push_back(num);
    }
    return std::vector<std::vector<int>>({newShape(in.at(0)->size(),v) });
}



template class Reshape<int>;
template class Reshape<float>;
template class Reshape<double>;