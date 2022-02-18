#include<string>
#include<memory>
#include<map>

#include"onnx.pb.h"
#include"../operator/Conv.h"
#include"../operator/Add.h"
#include"../operator/Clip.h"


//根据给定的att返回std::vector<int>
std::vector<int> att2int_vector(const onnx::AttributeProto& att);

//根据给定的att列表,返回一个可以通过name映射到对应的att的map
std::map<std::string,onnx::AttributeProto> attributeList2map( google::protobuf::RepeatedPtrField<onnx::AttributeProto> att);

//根据给定的node初始化Operator,并返回指针
template<typename T>
std::unique_ptr<Operator<T>> init_node(const onnx::NodeProto& node);

template<typename T>
std::unique_ptr<Tensor<T>> init_tensor(const onnx::TensorProto& tensor);