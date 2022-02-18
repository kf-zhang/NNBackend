#pragma once

#include <fstream>
#include <google/protobuf/util/time_util.h>
#include <iostream>
#include <string>
#include <map>

#include"onnx.pb.h"
#include"tool.hpp"
using namespace  std;


std::map<std::string,onnx::NodeProto> get_node_map(const onnx::ModelProto & model){
    std::map<std::string,onnx::NodeProto> m;
    for(auto node:model.graph().node()){
        m[ node.name() ] = node;
    }
    return m;
}
std::map<std::string,onnx::TensorProto > get_tensor_map(const onnx::ModelProto & model){
    std::map<std::string,onnx::TensorProto> m;
    for(auto tensor:model.graph().initializer() ){
        m[ tensor.name() ] = tensor;
    }
    return m;
}