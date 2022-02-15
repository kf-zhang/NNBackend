#include<testCommon/test.hpp>
#include<operator/Operator.h>
#include<operator/Add.h>

//测试Add的前向传播函数,无broadcast
void TEST_Add_AddForwardWIthoutBroadcast(){
    TEST_START

    std::vector<int> shape = {2,3};
    float data[6] = {0,1,2,3,4,5};
    Tensor<float> A(shape,data);
    Tensor<float> B(shape,data);
    Tensor<float> C(shape);

    Add<float> add;
    Operator<float>* op = &add;

    std::vector<Tensor<float>*> in({&A,&B});
    std::vector<Tensor<float>*> out({&C});

    add(in,out);
    auto p = C.cpu_pointer();
    for(int i=0;i<6;i++)
        assert(*(p.get()+i)==data[i]*2);
    

    (*op)(in,out);
    auto p1 = C.cpu_pointer();

    TEST_END
}


////使用虚类Operator调用Add的前向传播函数,无broadcast
void TEST_Add_OperatorForwardWIthoutBroadcast(){
    TEST_START

    std::vector<int> shape = {2,3};
    float data[6] = {0,1,2,3,4,5};
    Tensor<float> A(shape,data);
    Tensor<float> B(shape,data);
    Tensor<float> C(shape);

    Add<float> add;
    Operator<float>* op = &add;

    std::vector<Tensor<float>*> in({&A,&B});
    std::vector<Tensor<float>*> out({&C});

    (*op)(in,out);
    auto p = C.cpu_pointer();
    for(int i=0;i<6;i++)
        assert(*(p.get()+i)==data[i]*2);

    TEST_END
}


int main(){
    TEST_Add_AddForwardWIthoutBroadcast();
    TEST_Add_OperatorForwardWIthoutBroadcast();
    return 0;
}