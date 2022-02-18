#include<testCommon/test.hpp>
#include<operator/Operator.h>
#include<operator/Conv.h>


//测试默认参数下的卷积操作
void TEST_Conv_ConvForward(){

    TEST_START

    Conv<int> conv;

    int X_data[25] = {  0,1,2,3,4,
                        5,6,7,8,9,
                        10,11,12,13,14,
                        15,16,17,18,19,
                        20,21,22,23,24
                    };
    std::vector<int> X_shape = {1,1,5,5};

    int W_data[9] = {1,1,1,1,1,1,1,1,1};
    std::vector<int> W_shape = {1,1,3,3,};

    int B_data[1] = {0};
    std::vector<int> B_shape = {1};

    int exp_Y_data[9] = {54,63,72,99,108,117,144,153,162};
    std::vector<int> Y_shape = {1,1,3,3};

    Tensor<int> X(X_shape,X_data);
    Tensor<int> W(W_shape,W_data);
    Tensor<int> B(B_shape,B_data);
    Tensor<int> Y(Y_shape);

    conv( {&X,&W,&B} , {&Y} );

    auto p = Y.cpu_pointer();
    for(int i=0;i<9;i++)
        assert(exp_Y_data[i]==*(p.get()+i));

    TEST_END
}

//测试设置pads和strides后的卷积
void TEST_Conv_ConvForwardWithPaddingStrides(){
    TEST_START

    int X_data[35] = {  0,1,2,3,4,
                        5,6,7,8,9,
                        10,11,12,13,14,
                        15,16,17,18,19,
                        20,21,22,23,24,
                        25,26,27,28,29,
                        30,31,32,33,34
                    };
    std::vector<int> X_shape = {1,1,7,5};

    int W_data[9] = {1,1,1,1,1,1,1,1,1};
    std::vector<int> W_shape = {1,1,3,3,};

    int B_data[1] = {0};
    std::vector<int> B_shape = {1};

    int exp_Y_data[12] = {12,27,24,63,108,81,123,198,141,112,177,124};
    std::vector<int> Y_shape = {1,1,4,3};

    Tensor<int> X(X_shape,X_data);
    Tensor<int> W(W_shape,W_data);
    Tensor<int> B(B_shape,B_data);
    Tensor<int> Y(Y_shape);

    std::vector<int> pads = {1,1,1,1};
    std::vector<int> strides = {2,2};
    Conv<int> conv(W_shape,"",1,pads,strides);

    conv( {&X,&W,&B} , {&Y} );

    auto p = Y.cpu_pointer();
    for(int i=0;i<12;i++)
        assert(exp_Y_data[i]==*(p.get()+i));

    TEST_END
}

//测试只在一个维度上使用padding的卷积
void TEST_Conv_ConvForwardWithAsymmetricPadding(){
    TEST_START

    int X_data[35] = {  0,1,2,3,4,
                        5,6,7,8,9,
                        10,11,12,13,14,
                        15,16,17,18,19,
                        20,21,22,23,24,
                        25,26,27,28,29,
                        30,31,32,33,34
                    };
    std::vector<int> X_shape = {1,1,7,5};

    int W_data[9] = {1,1,1,1,1,1,1,1,1};
    std::vector<int> W_shape = {1,1,3,3,};

    int B_data[1] = {0};
    std::vector<int> B_shape = {1};

    int exp_Y_data[8] = {21,33,99,117,189,207,171,183};
    std::vector<int> Y_shape = {1,1,4,2};

    Tensor<int> X(X_shape,X_data);
    Tensor<int> W(W_shape,W_data);
    Tensor<int> B(B_shape,B_data);
    Tensor<int> Y(Y_shape);

    std::vector<int> pads = {1,0,1,0};
    std::vector<int> strides = {2,2};
    Conv<int> conv(W_shape,"",1,pads,strides);

    conv( {&X,&W,&B} , {&Y} );
    
    auto p = Y.cpu_pointer();
    for(int i=0;i<8;i++)
        assert(exp_Y_data[i]==*(p.get()+i));

    TEST_END
}

//测试 operator<<能否正常输出
void TEST_Conv_ConvPrint(){
    TEST_START

    std::vector<int> pads = {1,0,1,0};
    std::vector<int> strides = {2,2};
    std::vector<int> W_shape = {1,1,3,3,};
    Conv<int> conv(W_shape,"",1,pads,strides);
    std::cout<<conv;

    TEST_END
}

int main(){
    TEST_Conv_ConvForward();
    TEST_Conv_ConvForwardWithPaddingStrides();
    TEST_Conv_ConvForwardWithAsymmetricPadding();
    TEST_Conv_ConvPrint();
    return 0;
}