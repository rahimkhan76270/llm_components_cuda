cmake_minimum_required(VERSION 4.0)
project(llm_components_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(llm_components_cuda main.cu)

target_include_directories(llm_components_cuda PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
# Add all sources from your 'src' directory
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/*.cu
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cuh
        ${PROJECT_SOURCE_DIR}/src/*.h
)
target_sources(llm_components_cuda PRIVATE ${PROJECT_SOURCES})

set(CMAKE_CUDA_ARCHITECTURES 86)  # or 75, 86, etc.

add_subdirectory(external/cutlass)
include_directories(
        external/cutlass/include
        external/cutlass/tools/util/include
        external/cutlass/examples/common
)
#target_link_libraries(cutlass_exp PRIVATE cutlass)
include_directories(
        ${PROJECT_SOURCE_DIR}/external/cutlass/include
        ${PROJECT_SOURCE_DIR}/external/cutlass/tools/util/include
)
target_link_libraries(llm_components_cuda PRIVATE curand)
find_package(fmt)
target_link_libraries(llm_components_cuda PRIVATE fmt)
set_target_properties(llm_components_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
