cmake_minimum_required(VERSION 3.22.1)

project(SnuQS VERSION 1.1 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenMP REQUIRED)
find_package(ANTLR REQUIRED)
find_package(CUDAToolkit 11.7 REQUIRED)

add_subdirectory(src)
add_subdirectory(test)

add_subdirectory(python)
