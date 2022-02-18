#pragma once
#include"onnx.pb.h"

std::vector<int> att2int_vector(const onnx::AttributeProto& att){
    std::vector<int> v;
    for(int i=0;i<att.ints_size();i++)
        v.push_back( att.ints(i) );
    return v;
}
