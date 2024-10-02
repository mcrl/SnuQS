cmake_minimum_required(VERSION 3.29.0)

project(
  SnuQS
  VERSION 1.1
  LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OpenMP_FOUND)

find_package(CUDAToolkit 12.4 REQUIRED)
if(CUDAToolkit_FOUND)
  include_directories(BEFORE SYSTEM ${CUDAToolkit_INCLUDE_DIRS})

  execute_process(
    COMMAND bash "-c"
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u"
    OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES)
  string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  string(STRIP "${CMAKE_CUDA_ARCHITECTURES}" CMAKE_CUDA_ARCHITECTURES)
  string(REPLACE "\n" ";" CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif(CUDAToolkit_FOUND)

add_subdirectory(src)
