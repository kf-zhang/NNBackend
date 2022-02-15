#include"tensor/Tensor.hpp"
#include"testCommon/test.hpp"


void TEST_Tensor_ConstructorWithData(){
    TEST_START
    int data[6] = {0,1,2,3,4,5};
    std::vector<int> shape={3,2};
    Tensor<int> t(shape,data);

    std::unique_ptr<int> p = t.cpu_pointer();
    std::cout<<t<<"\n";
    
    assert(2==t.dim());
    assert(6==t.size());

    std::vector<int> t_shape = t.getShape();
    assert(shape==t_shape);

    for(int i=0;i<6;i++)
        assert(data[i]==*(p.get()+i) );
    TEST_END
}

void TEST_Tensor_ConstructorWithoutData(){
    TEST_START

    int *data = nullptr;
    std::vector<int> shape={};
    Tensor<int> t(shape,data);

    std::unique_ptr<int> p = t.cpu_pointer();
    std::cout<<t<<"\n";

    assert(nullptr==p);
    assert(0==t.dim());
    assert(0==t.size());
    std::vector<int> t_shape = t.getShape();
    assert(shape==t_shape);
    
    TEST_END
}

int main(){
    TEST_Tensor_ConstructorWithData();
    TEST_Tensor_ConstructorWithoutData();
}


