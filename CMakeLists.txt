cmake_minimum_required(VERSION 3.10)

project(NNBackend)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)

add_subdirectory(src)
list(APPEND EXTRA_LIBS operator)


add_executable(Tensor_test ./test/tensor/Tensor_test.cu)
target_link_libraries(Tensor_test PUBLIC ${EXTRA_LIBS})

add_executable(Add_test ./test/operator/Add_test.cu)
target_link_libraries(Add_test PUBLIC ${EXTRA_LIBS})