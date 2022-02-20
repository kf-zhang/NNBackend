#include<testCommon/test.hpp>
#include<operator/Operator.h>
#include<operator/BatchNormalization.h>

void TEST_BatchNormalization_BatchNormalizationOforward()
{
    TEST_START

    const int N = 2;
    const int C = 3;
    const int H = 2;
    const int W = 2;
    
    float X[N*C*H*W] = {0,1,2,3,4,5,6,7,8,9,10,11,
                        12,13,14,15,16,17,18,19,20,21,22,23};
    float mean[C] = {0,0,0};
    float var[C] = {1,1,1};
    float B[C] = {0,0,0};
    float scale[C] = {1,1,1};

    Tensor<float> tX({N,C,H,W},X);
    Tensor<float> tmean({C},mean);
    Tensor<float> tvar({C},var);
    Tensor<float> tB({C},B);
    Tensor<float> tscale({C},scale);
    Tensor<float> out({N,C,H,W});

    BatchNormalization<float> op;
    op({&tX,&tscale,&tB,&tmean,&tvar},{&out});

    std::cout<<out;

    TEST_END
}


int main()
{
    TEST_BatchNormalization_BatchNormalizationOforward();
}