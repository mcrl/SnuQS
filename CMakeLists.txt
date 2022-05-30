cmake_minimum_required(VERSION 3.10)

project(SnuQS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

set(SPDLOG_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/dep/spdlog-1.9.2/include/")
set(ANTLR4_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/dep/antlr4/runtime/Cpp/run/usr/local/include/antlr4-runtime")
set(BOOST_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/dep/boost")

set(LIBURING_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/dep/liburing/src/include")
set(LIBURING_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dep/liburing/src")


find_package(OpenMP)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(src)
