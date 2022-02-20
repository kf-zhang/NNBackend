#include<testCommon/test.hpp>
#include<operator/Operator.h>
#include<operator/GlobalAveragePool.h>


void TEST_GlobalAveragePool_Forward()
{
    TEST_START
    int X_data[25] = {  0,1,2,3,4,
                        5,6,7,8,9,
                        10,11,12,13,14,
                        15,16,17,18,19,
                        20,21,22,23,24
                    };
    std::vector<int> X_shape = {1,1,5,5};
    Tensor<int> X(X_shape,X_data);
    Tensor<int> Y({1,1,1,1});

    GlobalAveragePool<int> op;
    op({&X},{&Y});

    std::cout<<Y;

    TEST_END
}

int main()
{
    TEST_GlobalAveragePool_Forward();
    return 0;
}