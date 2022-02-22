#include <fstream>
#include <google/protobuf/util/time_util.h>
#include <iostream>
#include <string>
#include <map>

#include<graph/Graph.h>

#define N (9)
#define C (3)
#define H (32)
#define W (32)
#define BUF_SIZE (N*C*H*W)
#define OUT_SIZE (9*100)

//读取onnx文件,初始化model
int loadModel(int argc,char*argv[],onnx::ModelProto& model)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
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

int readData(const char* name,float* buf,int bufsize)
{
    std::ifstream in(name);
    
    int idx = 0;
    while (idx<bufsize &&in>>buf[idx] )
        idx++;
    in.close();
    return idx;
}




int main(int argc,char*argv[])
{
    onnx::ModelProto model;
    float input[BUF_SIZE];
    float output[OUT_SIZE];

    if(argc!=4)
    {
        std::cerr << "Usage: "<<argv[0]<<"onnxModelPath input output"<< std::endl;
    }
    
    loadModel(argc,argv,model);
    readData(argv[2],input,BUF_SIZE);
    readData(argv[3],output,OUT_SIZE);

    Graph<float> g(model);
    Tensor<float> in({N,C,H,W},input);
    auto out = g.forward({&in});
    out = g.forward({&in});


    auto result = out.at(0);
    int size = result.size();
    float maxErr = 0;
    auto p = result.cpu_pointer();
    std::cout<<"size: "<<size<<"\n";
    for(int i=0;i<size;i++)
    {   
        float err = abs( *(p.get()+i) - output[i] );
        // std::cout<<err<<" ";
        maxErr = maxErr>err?maxErr:err;
    }
    std::cout<<"max error"<<maxErr<<std::endl;

    return 0;
}
