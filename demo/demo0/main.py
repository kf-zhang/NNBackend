import torch
import torch.nn as nn
import onnxruntime
import onnx
import resnet

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.relu1 = nn.ReLU6()
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3)
        self.relu2 = nn.ReLU6()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self,X):
        X = self.relu1( self.conv1(X) ) 
        X = self.bn1(X)
        X = self.relu2(self.conv2(X))
        X = self.pool(X)
        return X

def writeTensor(X:torch.Tensor,fname:str):
    arr = X.reshape(-1)
    fd = open(fname,"w")
    for item in arr:
        fd.write( str(item)+" " ) 

    

if __name__ == "__main__":
    inputName = "input.txt"
    outputName = "output.txt"
    onnxFileName = "demo0.onnx"
    
    X = torch.randn([3,3,128,128])
    writeTensor(X.detach().numpy(),inputName)
    
    # model = net()
    model = resnet.resnet18()
    torch.onnx.export(
        model,
        X,
        onnxFileName,
        input_names = ["X"]   
    )
    
    model = onnx.load(onnxFileName)
    outputNames = [i.name for i in model.graph.output]

    
    sess = onnxruntime.InferenceSession(onnxFileName)
    outputs = sess.run(outputNames,{"X":X.detach().numpy()})
    writeTensor(outputs[0],outputName)
    print("input shape: ",X.detach().numpy().shape)
    print("output shape: ",outputs[0].shape)
    