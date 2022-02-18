#include<vector>
#include<map>
#include<string>
#include<memory>

#include"../onnx2graph/onnx.pb.h"

#include"../tensor/Tensor.hpp"
#include"../operator/Operator.h"
#include"../onnx2graph/onnx2graph.h"
template<typename T>
class Graph
{
    public:
        Graph(const onnx::ModelProto & model);
    private:
        std::vector< std::unique_ptr< Tensor<T> > > tensors;
        std::vector< std::unique_ptr< Operator<T> > > ops;
        
        std::map<std::string,int> name2tensorIdx;
        std::map<std::string,int> name2opIdx;
};


template<typename T>
Graph<T>::Graph(const onnx::ModelProto& model)
{
    int node_size = model.graph().node_size();
    auto nodes = model.graph().node();
    for(int i=0;i<node_size;i++)
        tensors.push_back( init_node( nodes.at(i) ) );
}