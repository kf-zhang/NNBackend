#include<cassert>

#include"Conv.h"
#include"../common/param.h"


template <typename T>
inline __device__ T GET(const T *p, int x0, int x1, int x2, int x3, int size0, int size1, int size2, int size3)
{
    bool valid = (x0 >= 0) && (x1 >= 0) && (x2 >= 0) && (x3 >= 0) && (x0 < size0) && (x1 < size1) && (x2 < size2) && (x3 < size3);
    int idx = x0 * (size1 * size2 * size3) + x1 * (size2 * size3) + x2 * (size3) + x3;
    return valid ? p[idx] : 0;
}

template <typename T>
inline __device__ void SET(T value, T *p, int x0, int x1, int x2, int x3, int size0, int size1, int size2, int size3)
{
    bool valid = (x0 >= 0) && (x1 >= 0) && (x2 >= 0) && (x3 >= 0) && (x0 < size0) && (x1 < size1) && (x2 < size2) && (x3 < size3);
    int idx = x0 * (size1 * size2 * size3) + x1 * (size2 * size3) + x2 * (size3) + x3;
    valid ? p[idx] = value : 0;
}

template <typename T>
__global__ void conv_kernel(const T *X,const T *W,const T *B, T *Y, int N, int C, int height, int width, int M, int out_height, int out_width, int kH, int kW,
                            int grid_width, int group, int stri0, int stri1, int pad0, int pad1, int pad2, int pad3, int dila0, int dila1)
{
    // X(N,C,height,width) W(M,C/group,kH,kW) B(M)
    // Y(N,M,out_height,out_width)
    int n = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.x;
    int w0 = threadIdx.y;
    int h_base = (blockIdx.z / grid_width) * BLOCK_SIZE;
    int w_base = (blockIdx.z % grid_width) * BLOCK_SIZE;
    int h = h0 + h_base;
    int w = w0 + w_base;

    int Cg = C / group;
    int Mg = M / group;

    T sum = 0;
    for (int c = 0; c < C / group; c++)
        for (int i = 0; i < kH; i++)
            for (int j = 0; j < kW; j++)
                sum += GET<T>(X, n, c + m / Mg * Cg, h * stri0 + i * dila0 - pad0, w * stri1 + j * dila1 - pad2, N, C, height, width) * GET<T>(W, m, c, i, j, M, Cg, kH, kW);
    SET<T>(sum + B[m], Y, n, m, h, w, N, M, out_height, out_width);
}

template<typename T>
std::ostream& operator<<(std::ostream&out,const Conv<T>& c)
{
    auto printVec = [&out] (std::string s,const std::vector<int>& v) 
    {
        out<<s<<":(";
        for(auto i:v)
            out << i <<",";
        out<<")\n";
    };
    printVec("kernel_shape",c.kernel_shape);
    printVec("pads",c.pads);
    printVec("strides",c.strides);
    printVec("dilations",c.dilations);
    out<<"auto pad str:"<<c.auto_pad<<"\n";
    return out;
}



//根据输入形状以及卷积核的形状计算输出的形状,X_shape应该为{N,C,H,W} W_shape应该为(M,C/group,kH,kW)
template<typename T>
std::vector<int> Conv<T>::output_shape(const std::vector<int> &X_shape, const std::vector<int> &W_shape) const
{
    std::vector<int> shape;
    assert(X_shape.size()==4);
    assert(W_shape.size()==4);

    int N = X_shape.at(0);
    int M = W_shape.at(0);

    int H = (X_shape.at(2) + pads.at(0) + pads.at(2) - dilations.at(0) * (W_shape.at(2) - 1) - 1) / strides.at(0) + 1;
    int W = (X_shape.at(3) + pads.at(1) + pads.at(3) - dilations.at(1) * (W_shape.at(3) - 1) - 1) / strides.at(1) + 1;

    shape.push_back(N);
    shape.push_back(M);
    shape.push_back(H);
    shape.push_back(W);
    return shape;
}



//初始化Conv算子,
//设置auto_pad不会对padding产生影响
//设置kernel_shape不会对运算产生影响，kernel shape取决于运行时的输入
template<typename T>
Conv<T>::Conv(
                const std::vector<int> &k_shape , 
                std::string pad_str, 
                int group_num, 
                const std::vector<int> &pad_ , 
                const std::vector<int> &stri, 
                std::vector<int> dila
            ) 
            :   auto_pad(pad_str),
                dilations(dila),
                group(group_num),
                kernel_shape(k_shape),
                pads(pad_),
                strides(stri) 
{

}

// in = {X,W,B} out = {Y};
//X shape为 (N,C,H,W) W shape为(M,C/group,kH,kW) B shape为(M) Y 形状为(N,M,newH,newH)
template<typename T>
void Conv<T>::operator()(const std::vector<Tensor<T>*> &in,const std::vector<Tensor<T>*> &out ) const
{   
    //检查 in 和 ou是否满足要求t
    assert( in.size()==3    );
    assert( out.size()==1   );

    Tensor<T>* X = in.at(0);
    Tensor<T>* W = in.at(1);
    Tensor<T>* B = in.at(2);
    Tensor<T>* Y = out.at(0);
    
    std::vector<int> X_shape = X->getShape();
    std::vector<int> W_shape = W->getShape();
    std::vector<int> B_shape = B->getShape();
    std::vector<int> Y_shape = Y->getShape();
    //检查X,W,B,Y的形状是否满足要求
    std::vector<int> O_shape = output_shape(X_shape,W_shape) ;
    assert( O_shape == Y_shape );
    assert(B_shape.size()==1);
    assert(B_shape.at(0)==Y_shape.at(1));


    dim3 block(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 grid(Y_shape.at(0), Y_shape.at(1), ceil(double(Y_shape.at(2)) / BLOCK_SIZE) * ceil(double(Y_shape.at(3)) / BLOCK_SIZE));
    int grid_width = ceil(double(Y_shape.at(3)) / BLOCK_SIZE);

     conv_kernel<T><<<grid, block>>>(
                X->gpu_pointer(), W->gpu_pointer(), B->gpu_pointer(), Y->raw_pointer(),
                X_shape.at(0), X_shape.at(1), X_shape.at(2), X_shape.at(3),
                Y_shape.at(1), Y_shape.at(2), Y_shape.at(3),
                W_shape.at(2), W_shape.at(3),
                grid_width,
                group,
                strides.at(0), strides.at(1),
                pads.at(0), pads.at(2), pads.at(1), pads.at(3),
                dilations.at(0), dilations.at(1)
                );
}

template<typename T>
std::vector<std::vector<int>> Conv<T>::outShape(const std::vector<Tensor<T>*> &in) const
{
    assert(in.size()>=2);
    std::vector<std::vector<int>> v;
    v.push_back(output_shape(in.at(0)->getShape(),in.at(1)->getShape() ));
    return  v;
}


template class Conv<int>;
template std::ostream& operator<<(std::ostream&, const Conv<int>&);

template class Conv<float>;
template std::ostream& operator<<(std::ostream&,const  Conv<float>&);

template class Conv<double>;
template std::ostream& operator<<(std::ostream&,const Conv<double>&);

