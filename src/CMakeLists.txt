enable_language(CUDA)

find_package(Protobuf REQUIRED)
add_library(
    onnx
    ./onnx2graph/onnx.pb.cc
)
target_link_libraries(
    onnx
    PUBLIC
    protobuf
)


add_library(
    operator
    ./operator/Add.cu
    ./operator/Conv.cu
    ./operator/Clip.cu
    ./operator/Gemm.cu
)
target_include_directories(
    operator
    INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)


add_library(
    onnx2graph
    ./onnx2graph/onnx2graph.cpp
)
target_link_libraries(
    onnx2graph
    PUBLIC
    onnx
    PUBLIC
    operator
)
target_include_directories(
    onnx2graph
    PUBLIC
    /usr/local/cuda/include
    INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)


add_library(
    graph
    ./graph/Graph.cpp
)
target_link_libraries(
    graph
    PUBLIC
    onnx2graph
)
target_include_directories(
    graph
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)