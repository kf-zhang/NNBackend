#include"tensor/Tensor.hpp"
#include"testCommon/test.hpp"


void TEST_Tensor_ConstructorWithData()
{
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

void TEST_Tensor_ConstructorWithoutData()
{
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

void TEST_Tensor_CopyContructor()
{
    TEST_START

    Tensor<int> t0({128,3,32,32});
    Tensor<int> t1(t0);

    auto p0 = t0.cpu_pointer();
    auto p1 = t1.cpu_pointer();

    for(int i=0;i<t0.size();i++)
        assert(*(p0.get()+i) ==*(p1.get()+i) );

    TEST_END
}

void TEST_Tensor_Assign()
{
    TEST_START

    Tensor<int> t0({128,3,32,32});
    auto p0 = t0.cpu_pointer();

    Tensor<int> t1;
    t1 = t0;
    auto p1 = t1.cpu_pointer();
    for(int i=0;i<t0.size();i++)
        assert(*(p0.get()+i) ==*(p1.get()+i) );

    Tensor<int> t2({1,2,3,4});
    t2 = t0;
    auto p2 = t2.cpu_pointer();
    for(int i=0;i<t0.size();i++)
        assert(*(p0.get()+i) ==*(p2.get()+i) );

    Tensor<int> t3({128,3,32,32});
    t3.setmem(t0);
    auto p3 = t3.cpu_pointer();
    for(int i=0;i<t0.size();i++)
        assert(*(p0.get()+i) ==*(p3.get()+i) );
    TEST_END
}


int main(){
    TEST_Tensor_ConstructorWithData();
    TEST_Tensor_ConstructorWithoutData();
    TEST_Tensor_CopyContructor();
    TEST_Tensor_Assign();
}


