ResNet18Model:=./demo/demo0/ResNet18.onnx
ResNet18Input:=./demo/demo0/ResNet18Input.txt
ResNet18Output:=./demo/demo0/ResNet18Output.txt

NetModel:=./demo/demo0/Net.onnx
NetInput:=./demo/demo0/NetInput.txt
NetOutput:=./demo/demo0/NetOutput.txt


runResNet:
	./build/demo0 $(ResNet18Model) $(ResNet18Input) $(ResNet18Output)

runNet:
	./build/demo0 $(NetModel) $(NetInput) $(NetOutput)
