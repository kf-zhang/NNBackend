from turtle import forward
import torch
import torch.nn as nn


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.relu1 = nn.ReLU6()
        self.conv2 = nn.Conv2d(16,32,3)
        self.relu2 = nn.ReLU6()
    def forward(self,X):
        X = self.relu1(self.conv1(X))
        X = self.relu2(self.conv2(X))
        return X

if __name__ == "__main__":
    X = torch.randn([128,3,32,32])
    model = net()
    
    torch.onnx.export(
        model,
        X,
        "demo0.onnx",
        input_names = ["X"]
    )
    
    out = model(X)
    print("out shape:",out.shape)