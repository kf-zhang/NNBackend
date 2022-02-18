#include <fstream>
#include <google/protobuf/util/time_util.h>
#include <iostream>
#include <string>
#include <map>

#include"onnx.pb.h"
#include "tool.hpp"
#include"./parse.hpp"
using namespace  std;

void disp_node(const onnx::NodeProto& node){
    std::cout<<"node name: "<<node.name()<<std::endl;
    std::cout<<"operator name: "<<node.op_type()<<std::endl;
    std::cout<<"node input: "<<std::endl;
    for(auto in:node.input())
        std::cout<<"\t"<< in <<std::endl;
    std::cout<<"node attribute name"<<std::endl;
    auto att = node.attribute();
    for(auto att:node.attribute() ){
        std::cout<<"\t"<< att.name() <<std::endl;
        std::cout<<"\t"<<onnx::AttributeProto_AttributeType_Name( att.type() ) <<std::endl;
        if(att.has_i())
            std::cout<<att.i() <<" ";
        else if(att.has_f())
            std::cout<<att.f()<<" ";
        else{
            std::vector<int> v(att2int_vector(att));
            for(auto i:v)
                std::cout<<i<<" ";
        }
        // int size = att.ints_size();
        // for(int i=0;i<size;i++)
            // std::cout<<att.ints(i)<<" ";
        std::cout<<std::endl;
    }
}

void disp_tensor(onnx::TensorProto& tensor){
    std::cout<<"tensor name: "<<tensor.name()<<std::endl;
    std::cout<<"tensor dim:(";
    std:vector<int> d(tensor.dims().begin(),tensor.dims().end());
    for(auto i:d )
        std::cout<<i<<",";
    std::cout<<")"<<std::endl;
    std::cout<<"tensor byte size: "<<tensor.ByteSizeLong()<<std::endl;
    std::cout<<"tensor data type: "<<onnx::TensorProto_DataType_Name( tensor.data_type() )<<std::endl;
    std::cout<<"tensor data size: "<<tensor.float_data_size()<<std::endl;
    
    std::cout<<"data location:"     <<tensor.DataLocation_Name(tensor.data_location() )<<std::endl;
    // float*p=tensor.mutable_float_data()->mutable_data();
    
    float*p=(float*)tensor.raw_data().data();
    int size = tensor.raw_data().size() /sizeof(float);
    std::cout<<"tensor data size:"<<size<<std::endl;
    // for(int i=0;i<size;i++)
        // std::cout<<p[i]<<" ";
}

std::vector<onnx::NodeProto> build_graph( google::protobuf::RepeatedPtrField<onnx::NodeProto> nodes){

}

void parse(const onnx::ModelProto & model){
    std::cout<<"start to parse model"<<std::endl;
    std::cout<< "model_size: "<<model.ByteSizeLong()<<" Bytes"<<std::endl;
    // model.graphs_size();
    auto graph = model.graph();
    
    auto nodes = graph.node();
    std::cout<<"node size: "<<nodes.size()<<std::endl;
    for(auto& node:nodes){
        if(node.op_type()=="Gemm")
            disp_node(node);
    }

    auto tensors = graph.initializer();
    
    // for(auto t:tensors)
        // std::cout<<t.name()<<std::endl;
    // for(auto tensor:tensors)
            // disp_tensor(tensor);
    

    
    std::cout<<"finish parsing model"<<std::endl;
}

int main(int argc,char*argv[]){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    if (argc != 2) {
        cerr << "Usage:  " << argv[0] << " ONNX_FILE" << endl;
        return -1;
    }

    
    onnx::ModelProto model;
    {
        fstream input(argv[1], ios::in | ios::binary);
        if (!model.ParseFromIstream(&input)) {
            cerr << "Failed to parse address book." << endl;
            return -1;
        }
    }

    parse(model);


    google::protobuf::ShutdownProtobufLibrary();

    return 0;


}