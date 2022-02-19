#include<vector>
#include<map>
#include<string>
#include<memory>
#include<stack>
#include<set>

#include"../onnx2graph/onnx.pb.h"

#include"../tensor/Tensor.hpp"
#include"../operator/Operator.h"
#include"../onnx2graph/onnx2graph.h"


template<typename T>
class Graph
{
    public:
        template<typename dataT>
        friend std::ostream& operator<<(std::ostream& os,const Graph<dataT>& g);
        Graph(const onnx::ModelProto & model);
        Graph() = default;
        std::vector<Tensor<T>> forward(const std::vector<Tensor<T>*> in);
        bool setTensor(const std::vector<std::string>&names,const std::vector<Tensor<T>*> val);
        std::vector<Tensor<T>> fetchTensors(const std::vector<std::string>& fetchTensorNames );
    protected:
        std::string opIdx2name(int idx) const;
        std::vector<int> forwardOrder(const onnx::ModelProto& model);
        void initTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto>& initializers);
        void initOperator(const google::protobuf::RepeatedPtrField<onnx::NodeProto>& nodes);
        bool runOp(int idx);
    private:
        std::vector< std::unique_ptr< Tensor<T> > > tensors;//计算图中的tensor
        std::vector< std::unique_ptr< Operator<T> > > ops;//计算图中的operator
        std::vector< std::pair<std::vector<std::string>,std::vector<std::string>> > opInOut;//每个operator输入和输出tensor的名字
        std::map<std::string,int> name2tensorIdx;//名字到tensor下标的映射
        std::map<std::string,int> name2opIdx;//名字到operator下标的映射
        std::vector<int> forwardPath;//计算图正向传播的拓扑排序
        std::set<std::string> initialized;//在onnx文件中已经初始化过的tensor的名字
        std::vector<std::string> inputs;//输入tensor的名字
        std::vector<std::string> outputs;//输出tensor的名字
};