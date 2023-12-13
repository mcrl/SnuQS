cmake_minimum_required(VERSION 3.22.1)

project(
  SnuQS
  VERSION 2
  LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OpenMP_FOUND)

find_package(CUDAToolkit 11.7 REQUIRED)
if(CUDAToolkit_FOUND)
  include_directories(BEFORE SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
endif(CUDAToolkit_FOUND)

include_directories(BEFORE SYSTEM include)

add_subdirectory(src)
add_subdirectory(tests)
