#include <fstream>
#include <google/protobuf/util/time_util.h>
#include <iostream>
#include <string>
#include <map>

#include<testCommon/test.hpp>
#include<operator/Operator.h>
#include<operator/Conv.h>
#include<onnx2graph/onnx2graph.h>


//测试能否根据node生成对应的operator
void TEST_onnx2graph_intinodeConv(const onnx::NodeProto &node){
    TEST_START

    std::unique_ptr<Operator<int>> p = init_node<int>(node);
    Conv<int> *pConv = static_cast<Conv<int>*>(p.get());
    std::cout<<*pConv;
    TEST_END
}

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

//从model中读取所有类型为op_type的算子
std::vector<onnx::NodeProto> getNodeByType(const onnx::ModelProto& model,const std::string& op_type)
{
    auto graph = model.graph();
    auto nodes = graph.node();

    std::vector<onnx::NodeProto> v;
    for(auto& node:nodes)
        if(node.op_type()== op_type)
            v.push_back(node);
    return v;
}


int main(int argc,char*argv[])
{
    onnx::ModelProto model;
    if( loadModel(argc,argv,model) )
        return -1;
    
    auto conv_ops = getNodeByType(model,"Conv");
    TEST_onnx2graph_intinodeConv(conv_ops.at(0));


    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}