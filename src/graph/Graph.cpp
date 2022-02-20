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
    
    for(const auto &val:model.graph().input())
        inputs.push_back(val.name());
    for(const auto&val:model.graph().output() )
        outputs.push_back(val.name());

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
//如果tensor还未分配内存,先给tensor分配内存,再拷贝
//如果已经分配内存,拷贝时需要确保形状相同
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
        int tensorIdx = name2tensorIdx[name];
        if( tensorIdx == 0 )//graph中不存在对应的tensor
        {
            std::cerr<<"invalid tensor name: "<<name<<"\n";
            return false;
        }
        
        if( !tensors[tensorIdx] )//graph中tensor未分配内存
        {
            tensors[tensorIdx] = std::unique_ptr< Tensor<T> >(new Tensor<T>(p->getShape()) );
        }
        if( tensors[tensorIdx].get() ->setmem(*p) == false )//拷贝并检查返回值是否正确
        {
            std::cerr<<"setmem error in setTensor\n";
            return false;
        }
    }
    return true;
}

//运行下标为idx的算子,输入输出来源于graph的tensor中
//输入tensor必须已经初始化完成
//输出tensor如果已经初始化,但是形状与计算所得形状不一致,返回false
//如果输出tensor未初始化,则初始化该tensor并进行计算
template<typename T>
bool Graph<T>::runOp(int opIdx)
{
    if( opIdx<0 || opIdx>=ops.size() )
    {
        std::cerr<<"invalid idx "<<opIdx<<" in runOp"<<"\n";
        return false;
    }
    std::vector<Tensor<T>*> inputTensorPointers;
    std::vector<Tensor<T>*> outputTensorPointers;

    std::cout<<"run op: "<<opIdx2name(opIdx)<<"\n";
    std::vector<std::vector<int>> inputTensorShapes;
    for( const std::string& inputTensorName: opInOut.at(opIdx).first )
    {
        int inputTensorIdx = name2tensorIdx[inputTensorName];
        if(!tensors.at(inputTensorIdx))//输入tensor未初始化
            return false;
        inputTensorShapes.push_back( tensors.at(inputTensorIdx).get()->getShape() );
        inputTensorPointers.push_back( tensors.at(inputTensorIdx).get() );

        std::cout<<"\t input name: "<<inputTensorName<<"\n";
    }

    std::vector<std::vector<int>> outTensorShapes = ops.at(opIdx).get()->outShape(inputTensorShapes);


    int outTensorNum = outTensorShapes.size();
    for(int i=0;i<outTensorNum;i++)
    {
        int outTensorIdx = name2tensorIdx[ opInOut[opIdx].second[i] ]  ; 
        if( ! tensors.at(outTensorIdx) )//输出tensor还没有分配内存
        {
            std::cout<<"\t allocate space for operator "<<opIdx2name(opIdx)<<"\n";
            tensors.at(outTensorIdx) = std::unique_ptr<Tensor<T>>( new Tensor<T>(outTensorShapes.at(i)) );
        }

        outputTensorPointers.push_back( tensors.at(outTensorIdx).get() );

        std::cout<<"\t output name: "<<opInOut[opIdx].second[i]<<"\n";
    }
    
    ops[opIdx]->operator()(inputTensorPointers,outputTensorPointers);
    return true;
}

//从计算图中取出fetchTensorNames中对应的tensor
//如果计算图中不存在对应的tensor,则返回形状为{0}的tensor
template<typename T> 
std::vector<Tensor<T>> Graph<T>::fetchTensors(const std::vector<std::string> &fetchTensorNames)
{
    std::vector<Tensor<T>> v;
    for(const auto& name:fetchTensorNames)
    {
        Tensor<T>* tensorPointer = tensors[ name2tensorIdx[name] ].get();
        if( tensorPointer )
            v.push_back(*tensorPointer);
        else
            v.push_back(Tensor<T>({0}));
    }
    return v;
}


//计算图正向传播
//首先按照in初始化inputs对应的tensor
//然后按照拓扑排序逐个运行 operator
//最后按照outputs取出计算图中对应的tensor并返回
template<typename T>
std::vector<Tensor<T>> Graph<T>::forward(const std::vector<Tensor<T>*> in)
{
    
    if( !setTensor(inputs,in) )
    {
        std::cerr<<"fail to set input tensors in forward"<<std::endl;
        return std::vector<Tensor<T>>();
    }

    for(int idx:forwardPath )
    {
        if( !runOp(idx) )
        {
            std::cerr<<"fail to run operator "<<opIdx2name(idx)<<" in forward"<<std::endl;
            return std::vector<Tensor<T>>();
        }
    }

    return fetchTensors(outputs);
}



template class Graph<int>;
template std::ostream& operator<<(std::ostream& os,const Graph<int>& g);

template class Graph<float>;
template std::ostream& operator<<(std::ostream& os,const Graph<float>& g);

template class Graph<double>;
template std::ostream& operator<<(std::ostream& os,const Graph<double>& g);