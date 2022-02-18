parse:parse.cpp onnx.pb.cc onnx.pb.h
	pkg-config --cflags protobuf  # fails if protobuf is not installed
	c++ -std=c++11 parse.cpp onnx.pb.cc -o parse `pkg-config --cflags --libs protobuf`
run:parse
	./parse ./mobilenet_v2.onnx