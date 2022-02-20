#pragma once
#include"./Operator.h"

template<typename T>
class Conv: public Operator<T,T>
{
    private:
        std::string auto_pad;
        std::vector<int> dilations;
        int group;
        std::vector<int> kernel_shape;
        std::vector<int> pads;
        std::vector<int> strides;   
    public:
        template<typename A>
        friend std::ostream& operator<<(std::ostream&out,const Conv<A>& c);

        std::vector<int> output_shape(const std::vector<int> &X_shape, const std::vector<int> &W_shape) const;
        std::vector<std::vector<int>> outShape(const std::vector<Tensor<T>*> &in) const override;
        Conv(   const std::vector<int> &k_shape = {}, 
                std::string pad_str = "", 
                int group_num = 1, 
                const std::vector<int> &pad_ = {0, 0, 0, 0}, 
                const std::vector<int> &stri = {1, 1}, 
                std::vector<int> dila = {1, 1}
            );                                                  
        void operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const override;
};