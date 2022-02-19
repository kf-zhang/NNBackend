#include <fstream>
#include <google/protobuf/util/time_util.h>
#include <iostream>
#include <string>
#include <map>


#include<testCommon/test.hpp>
#include<graph/Graph.h>


//读取onnx文件,初始化model
int loadModel(int argc,char*argv[],onnx::ModelProto& model)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    if (argc != 2) 
    {
        std::cerr << "Usage:  " << argv[0] << " ONNX_FILE" << std::endl;
        return -1;
    }
    {
        std::fstream input(argv[1], std::ios::in | std::ios::binary);
        if (!model.ParseFromIstream(&input)) 
        {
            std::cerr << "Failed to parse onnx file." << std::endl;
            return -1;
        }
    }
    return 0;
}


void TEST_Graph_GraphInit(const onnx::ModelProto& model)
{
    TEST_START

    Graph<int> g(model);
    std::cout << g<<"\n";
    TEST_END
}

void TEST_Graph_GraphForward(const onnx::ModelProto& model)
{
    TEST_START

    Graph<int> g(model);
    Tensor<int> input0({1,3,8,8});
    Tensor<int> input1({1,3,9,9});
    g.forward( {&input0} );
    g.forward( {&input1} );
    TEST_END
}


int main(int argc,char*argv[])
{
    onnx::ModelProto model;
    if( loadModel(argc,argv,model) )
        return -1;
    
    TEST_Graph_GraphInit(model);

    TEST_Graph_GraphForward(model);
    return 0;
}

