#include"Graph.h"

//打印算子的拓扑排序
template<typename T>
std::ostream& operator<<(std::ostream& os,const Graph<T>& g)
{
    for(int i:g.forwardPath)
        os<< g.opIdx2name(i)<<" ";
    return os;
}

//根据算子的下标返回算子的名称
template<typename T>
std::string Graph<T>::opIdx2name(int idx) const
{
    for(auto p:name2opIdx)
        if(p.second==idx)
            return p.first;
    return std::string("");
}


//根据model.graph().initializer()初始化tensor
template<typename T>
void Graph<T>:: initTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> & initializers)
{
    //tensors[0]将会保留,不会真正分配内存,用于检查错误
    tensors.push_back( std::unique_ptr<Tensor<T>>(nullptr));
    name2tensorIdx["__FAKE__TENSOR__"] = 0;

    int idx = 1;
    for(const auto& tensor:initializers)
    {
        tensors.push_back( init_tensor<T>(tensor) );
        name2tensorIdx[tensor.name()] = idx;
        idx++;
        initialized.insert( tensor.name());
    }
}

//根据nodes初始化算子
template<typename T>
void Graph<T>::initOperator(const google::protobuf::RepeatedPtrField<onnx::NodeProto>& nodes)
{
    int node_size = nodes.size();
    for(int i=0;i<node_size;i++)
    {
        auto node = nodes.at(i);
        auto inputs = node.input();
        auto outputs = node.output();
        //初始化算子,并记录算子名称与算子下标的映射
        ops.push_back( init_node<T>( node ) );
        name2opIdx[nodes.at(i).name()] = i;
        for(const auto& name:inputs )
        {
            //不在initializer的tensor,以空指针初始化，并记录tensor名称与tensor下标的映射
            if( !name2tensorIdx[name] )
            {
                tensors.push_back( std::unique_ptr< Tensor<T> >(nullptr) ) ;
                name2tensorIdx[name] = tensors.size()-1;
            }
        }
        for(const auto& name:outputs )
        {
            //不在initializer的tensor,以空指针初始化，并记录tensor名称与tensor下标的映射
            if( !name2tensorIdx[name] )
            {
                tensors.push_back( std::unique_ptr< Tensor<T> >(nullptr) ) ;
                name2tensorIdx[name] = tensors.size()-1;
            }
        }
    }

    for(int i=0;i<node_size;i++)
    {
        auto node = nodes.at(i);
        auto inputs = node.input();
        auto outputs = node.output();

        std::vector<std::string> inV;
        std::vector<std::string> outV;

        for(const auto&name:inputs)
            inV.push_back(name);

        for(const auto&name:outputs)
            outV.push_back(name);
        
        opInOut.push_back(std::pair<std::vector<std::string>,std::vector<std::string>>(inV,outV));

    }
}


//给定model文件,初始化graph,包括operator的初始化,
template<typename T>
Graph<T>::Graph(const onnx::ModelProto& model)
{
    initTensor(model.graph().initializer());
    initOperator(model.graph().node());
    forwardPath = forwardOrder(model);
    
    for(const auto &name:model.graph().input())
        inputs.push_back(name);
    for(const auro&name:model.graph().input())
        outputs.push_back(name);

    std::cout<<"the number of operator in onnx : "<<model.graph().node_size()<<"\n";
    std::cout<<"the number of operator in forwardPath : "<<forwardPath.size()<<"\n";
}


//根据拓扑排序得到算子的执行顺序
template<typename T>
std::vector<int> Graph<T>::forwardOrder(const onnx::ModelProto& model)
{
    auto nodes = model.graph().node();
    int node_size = model.graph().node_size();
    std::map<std::string,std::vector<int>> tensor2node;
    std::vector<int> inDegree(ops.size());

    for(int i=0;i<node_size;i++)
    {
        auto node = nodes.at(i);
        int pre = 0;
        for( const auto& name :node.input() )
        {
            tensor2node[name].push_back( name2opIdx[node.name()] );
            pre+=( initialized.find(name)==initialized.end() );
        }
        inDegree[name2opIdx[node.name()] ] = pre;
    }

    std::stack<int> s;
    std::vector<int> result;
    for(auto val:model.graph().input() )
        inDegree[name2opIdx[val.name()]] = 0;
    
    for(int i=0;i<ops.size();i++)
        if( !inDegree[i] )
            s.push(i);
    while(!s.empty())
    {
        int idx = s.top();
        s.pop();
        result.push_back(idx);

        for(auto out:opInOut[idx].second)
        {
            for(int i : tensor2node[out] )
            {
                inDegree[ i  ]--;
                if(!inDegree[ i ])
                    s.push( i );
            }
        }
    }
    return result;

}

//将名字为names的tensor设置为val
template<typename T>
bool Graph<T>::setTensor(const std::vector<std::string>&names,const std::vector<Tensor<T>*> val)
{
    if( names.size() != val.size() )
        return false;
    int N = names.size();
    for(int i=0;i<N;i++)
    {
        std::string name = names.at(i);
        Tensor<T>* p = val.at(i);
        int idx = name2tensorIdx[name];
        if( idx == 0 )
        {
            std::cerr<<"invalid tensor name: "<<name<<"\n";
            return false;
        }
        if( !( tensors.at(idx).get() )->setmem(*p) ) 
            return false;
    }
    return true;
}

//运行下标为idx的算子,输入输出来源于graph的tensor中
template<typename T>
bool Graph<T>::runOp(int idx)
{
    if(idx<0||idx>=ops.size()){
        std::cerr<<"invalid idx "<<idx<<" in runOp"<<"\n";
        return false;
    }
}

//计算图正向传播
template<typename T>
std::vector<Tensor<T>> Graph<T>::forward(const std::vector<Tensor<T>*> in)
{
    if( !setTensor(inputs,in) )
        return std::vector<Tensor<T>>();
    for(int idx:forwardPath )
        if( !runOp(idx) )
            return false;
}



template class Graph<int>;
template std::ostream& operator<<(std::ostream& os,const Graph<int>& g);

template class Graph<float>;
template std::ostream& operator<<(std::ostream& os,const Graph<float>& g);

template class Graph<double>;
template std::ostream& operator<<(std::ostream& os,const Graph<double>& g);