cmake_minimum_required(VERSION 3.10)

project(NNBackend)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)

add_subdirectory(src)
list(APPEND EXTRA_LIBS operator)
list(APPEND EXTRA_LIBS onnx2graph)
list(APPEND EXTRA_LIBS graph)

add_executable(Tensor_test ./test/tensor/Tensor_test.cu)
target_link_libraries(Tensor_test PUBLIC ${EXTRA_LIBS})

add_executable(Add_test ./test/operator/Add_test.cu)
target_link_libraries(Add_test PUBLIC ${EXTRA_LIBS})

add_executable(Conv_test ./test/operator/Conv_test.cu)
target_link_libraries(Conv_test PUBLIC ${EXTRA_LIBS})

add_executable(Clip_test ./test/operator/Clip_test.cu)
target_link_libraries(Clip_test PUBLIC ${EXTRA_LIBS})

add_executable(onnx2graph_test ./test/onnx2graph/onnx2graph_test.cu)
target_link_libraries(onnx2graph_test PUBLIC ${EXTRA_LIBS})

add_executable(Graph_test ./test/graph/Graph_test.cpp)
target_link_libraries(Graph_test PUBLIC ${EXTRA_LIBS})

