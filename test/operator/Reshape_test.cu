#include<testCommon/test.hpp>
#include<operator/Operator.h>
#include<operator/Reshape.h>



void TEST_Reshape_Reshape()
{
    TEST_START

    int data0[24] ={0,44,92,214,4324,54,62346,76,1,15,657,153,647,17635,123};
    Tensor<int> A({1,2,3,4},data0);
    Tensor<int> B({1*2*3*4});
    

    int data[4]={1,2,3,-1};
    Tensor<int> newShape({4},data);
    Reshape<int> op;

    op({&A,&newShape},{&B});

    auto p1 = A.cpu_pointer();
    auto p2 = B.cpu_pointer();

    int size = A.size();
    for(int i=0;i<size;i++)
        assert( *(p1.get()+i)==*(p2.get()+i) );


    TEST_END
}

int main()
{
    TEST_Reshape_Reshape();
    return 0;
}