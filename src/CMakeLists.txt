
enable_language(CUDA)
add_library(
    operator
    ./operator/Add.cu
    ./operator/Conv.cu
)

target_include_directories(
    operator
    INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)