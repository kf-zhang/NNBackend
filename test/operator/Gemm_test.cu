#include<testCommon/test.hpp>
#include<operator/Operator.h>
#include<operator/Gemm.h>


void TEST_Gemm_Kernel0()
{
    TEST_START

    int A_data[8] = {0,1,2,3,4,5,6,7};
    std::vector<int> A_shape={2,4};
    Tensor<int> A(A_shape,A_data);

    int B_data[8] = {0,1,2,3,4,5,6,7};
    std::vector<int> B_shape={4,2};
    Tensor<int> B(B_shape,B_data);


    int C_data[4] = {0,1,2,3};
    std::vector<int> C_shape={2,2};
    Tensor<int> C(C_shape,C_data);

    
    std::vector<int> Y_shape={2,2};
    Tensor<int> Y(Y_shape);

    Gemm<int> op1;
    op1({&A,&B},{&Y});

    int expY_data1[4] = {28,34,76,98};
    auto p1 = Y.cpu_pointer();
    for(int i=0;i<Y.size();i++)
        assert(  *(p1.get()+i) == expY_data1[i] );

    TEST_END

}


void TEST_Gemm_Kernel1()
{
    TEST_START

    int A_data[8] = {0,1,2,3,4,5,6,7};
    std::vector<int> A_shape={2,4};
    Tensor<int> A(A_shape,A_data);

    int B_data[8] = {0,1,2,3,4,5,6,7};
    std::vector<int> B_shape={4,2};
    Tensor<int> B(B_shape,B_data);


    int C_data[4] = {0,1,2,3};
    std::vector<int> C_shape={2,2};
    Tensor<int> C(C_shape,C_data);

    
    std::vector<int> Y_shape={2,2};
    Tensor<int> Y(Y_shape);

    Gemm<int> op0;
    op0({&A,&B,&C},{&Y});


    int expY_data0[4] = {28,35,78,101};
    auto p0 = Y.cpu_pointer();
    for(int i=0;i<Y.size();i++)
        assert(  *(p0.get()+i) == expY_data0[i] );


    TEST_END   
}

//测试
void TEST_Gemm_Kernel2()
{
    TEST_START

    int A_data[8] = {0,1,2,3,4,5,6,7};
    std::vector<int> A_shape={2,4};
    Tensor<int> A(A_shape,A_data);

    int B_data[8] = {0,1,2,3,4,5,6,7};
    std::vector<int> B_shape={4,2};
    Tensor<int> B(B_shape,B_data);


    int C_data[2] = {7,8};
    std::vector<int> C_shape={2,1};
    Tensor<int> C(C_shape,C_data);

    
    std::vector<int> Y_shape={2,2};
    Tensor<int> Y(Y_shape);

    Gemm<int> op0;
    op0({&A,&B,&C},{&Y});


    int expY_data0[4] = {35,41,84,106};
    auto p0 = Y.cpu_pointer();
    for(int i=0;i<Y.size();i++)
        assert(  *(p0.get()+i) == expY_data0[i] );


    Gemm<int> op1(1.0,2.0);
    op1({&A,&B,&C},{&Y});

    int expY_data1[4] = {42,48,92,114};
    auto p1 = Y.cpu_pointer();
    for(int i=0;i<Y.size();i++)
        assert(  *(p1.get()+i) == expY_data1[i] );

    TEST_END   
}



void TEST_Gemm_Kernel3()
{
    TEST_START

    int A_data[8] = {0,1,2,3,4,5,6,7};
    std::vector<int> A_shape={2,4};
    Tensor<int> A(A_shape,A_data);

    int B_data[8] = {0,1,2,3,4,5,6,7};
    std::vector<int> B_shape={4,2};
    Tensor<int> B(B_shape,B_data);


    int C_data[2] = {7,8};
    std::vector<int> C_shape={1,2};
    Tensor<int> C(C_shape,C_data);

    
    std::vector<int> Y_shape={2,2};
    Tensor<int> Y(Y_shape);

    Gemm<int> op0;
    op0({&A,&B,&C},{&Y});


    int expY_data0[4] = {35,42,83,106};
    auto p0 = Y.cpu_pointer();
    for(int i=0;i<Y.size();i++)
        assert(  *(p0.get()+i) == expY_data0[i] );


    Gemm<int> op1(1.0,2.0);
    op1({&A,&B,&C},{&Y});

    int expY_data1[4] = {42,50,90,114};
    auto p1 = Y.cpu_pointer();
    for(int i=0;i<Y.size();i++)
        assert(  *(p1.get()+i) == expY_data1[i] );

    TEST_END   
}


int main(int argc,char*argv[])
{
    TEST_Gemm_Kernel0();
    TEST_Gemm_Kernel1();
    TEST_Gemm_Kernel2();
    TEST_Gemm_Kernel3();
}