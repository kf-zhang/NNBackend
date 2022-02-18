#include<testCommon/test.hpp>
#include<operator/Operator.h>
#include<operator/Clip.h>


////使用虚类Operator调用Clip的前向传播函数
void TEST_Clip_OperatorForward(){
    TEST_START

    std::vector<int> shape = {2,3};
    int data[6] = {-3,1,9,2,7,0};
    Tensor<int> A(shape,data);
    int min = 0;
    int max = 6;
    Tensor<int> Tmax({1},&max);
    Tensor<int> Tmin({1},&min);
    
    int exp[6] = {0,1,6,2,6,0};;
    Tensor<int> C(shape);

    Clip<int> clip;
    Operator<int>* op = &clip;
    
    std::vector<Tensor<int>*> in({&A,&Tmin,&Tmax});
    std::vector<Tensor<int>*> out({&C});
    
    (*op)(in,out);
    auto p = C.cpu_pointer();
    for(int i=0;i<6;i++)
        assert(*(p.get()+i)==exp[i]);

    TEST_END
}

int main(){
    TEST_Clip_OperatorForward();
}