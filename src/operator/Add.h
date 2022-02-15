template<typename T>
class Add:public Operator<T,T>
{
public:
    void operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const override;
};

// template class Add<int>;
// template class Add<float>;
// template class Add<double>;