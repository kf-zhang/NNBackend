#include"onnx2graph.h"

//根据给定的att返回std::vector<int>
std::vector<int> att2int_vector(const onnx::AttributeProto& att)
{
    std::vector<int> v;
    for(int i=0;i<att.ints_size();i++)
        v.push_back( att.ints(i) );
    return v;
}

//根据给定的att列表,返回一个可以通过name映射到对应的att的map
std::map<std::string,onnx::AttributeProto> attributeList2map( google::protobuf::RepeatedPtrField<onnx::AttributeProto> att)
{
    std::map<std::string,onnx::AttributeProto> m;
    for(const auto& item:att)
        m[item.name()] = item;
    return m;
}


//根据给定的node初始化Operator,并返回指针
template<typename T>
std::unique_ptr<Operator<T>> init_node(const onnx::NodeProto& node)
{

    if( node.op_type() == "Conv")
    {
        auto m = attributeList2map( node.attribute() );
        return std::unique_ptr<Operator<T>>( 
                                                new Conv<T>
                                                (
                                                    att2int_vector( m["kernel_shape"] ),
                                                    "",/*由于Conv未实现auto_pad,所以使用空字符串初始化 m["auto_pad"].s(),*/ 
                                                    m["group"].i(),
                                                    att2int_vector( m["pads"]),
                                                    att2int_vector( m["strides"]),
                                                    att2int_vector( m["dilations"])
                                                ) 
                                            );
    }
    if( node.op_type() == "Add")
        return std::unique_ptr<Operator<T>>( new Add<T>() );
    if( node.op_type() == "Clip")
        return std::unique_ptr<Operator<T>>( new Clip<T>() );
    std::cout<<"unimplementated op_type "<< node.op_type() << " in init_node\n";
    return std::unique_ptr<Operator<T>>( nullptr );
}


//根据给定的 const onnx::TensorProto& 类型的tensor
//初始化Tensor<T>类型的tensor,并返回指针
template<typename T>
std::unique_ptr<Tensor<T>> init_tensor(const onnx::TensorProto& tensor)
{
    auto dim = tensor.dims();
    auto p = tensor.raw_data().data();
    return std::unique_ptr<Tensor<T>>(
                                        new Tensor<T>
                                        (
                                            std::vector<int>( dim.begin(),dim.end() ),
                                            (T*)p
                                        )
                                      );
}


template std::unique_ptr<Operator<int>> init_node(const onnx::NodeProto& node);
template std::unique_ptr<Operator<float>> init_node(const onnx::NodeProto& node);
template std::unique_ptr<Operator<double>> init_node(const onnx::NodeProto& node);

template std::unique_ptr<Tensor<int>> init_tensor(const onnx::TensorProto& tensor);
template std::unique_ptr<Tensor<float>> init_tensor(const onnx::TensorProto& tensor);
template std::unique_ptr<Tensor<double>> init_tensor(const onnx::TensorProto& tensor);