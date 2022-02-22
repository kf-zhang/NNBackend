import torch
import torch.nn as nn
import onnxruntime
import onnx
import resnet

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,stride=2)
        self.relu1 = nn.ReLU6()
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16,32,1)
        self.relu2 = nn.ReLU6()
        
        self.conv3 = nn.Conv2d(32,192,1,groups=32)
        self.relu3 = nn.ReLU6()
        
        self.conv4 = nn.Conv2d(192,32,1,groups=32)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self,X):
        X = self.relu1( self.conv1(X) ) 
        X = self.bn1(X)
        X = self.relu2(self.conv2(X))
        
        X = self.relu3( self.conv3(X) )
        X = self.conv4(X) 
        X = self.pool(X)
        X = X.view(X.size(0),-1)
        return X

def writeTensor(X:torch.Tensor,fname:str):
    arr = X.reshape(-1)
    fd = open(fname,"w")
    for item in arr:
        fd.write( str(item)+" " ) 

def getNet():
    inputName = "NetInput.txt"
    outputName = "NetOutput.txt"
    onnxFileName = "Net.onnx"
    
    
    X = torch.randn([9,3,32,32])
    writeTensor(X.detach().numpy(),inputName)
    
    model = net()
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
    print("net input shape: ",X.detach().numpy().shape)
    print("net output shape: ",outputs[0].shape)    

def getResNet():
    inputName = "ResNet18Input.txt"
    outputName = "ResNet18Output.txt"
    onnxFileName = "ResNet18.onnx"
    
    
    X = torch.randn([9,3,32,32])
    writeTensor(X.detach().numpy(),inputName)
    
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
    print("resnet18 input shape: ",X.detach().numpy().shape)
    print("resnet18 output shape: ",outputs[0].shape)


if __name__ == "__main__":
    getNet()
    getResNet()
    