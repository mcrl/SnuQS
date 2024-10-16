execute_process(COMMAND python3 -m pybind11 --includes
                OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

execute_process(COMMAND python3-config --extension-suffix
                OUTPUT_VARIABLE LIBRARY_SUFFIX)
string(STRIP "${LIBRARY_SUFFIX}" LIBRARY_SUFFIX)
include_directories("${PYBIND_INCLUDE_DIRS}")

add_library(
  _C SHARED
  pymodule.cc
  core/runtime.cc
  core/cuda/runtime.cu
  functionals/functionals.cc
  operation/operation.cc
  operation/gate_operations.cc
  operation/gate_operations_impl_cpu.cc
  operation/gate_operations_impl_cuda.cu
  operation/initialization_impl_cpu.cc
  operation/initialization_impl_cuda.cu
  buffer/buffer_cpu.cc
  buffer/buffer_pinned.cc
  buffer/buffer_cuda.cc
  buffer/buffer_storage.cc
  buffer/hier_buffer.cc
  result_types/state_vector.cc)
set_target_properties(_C PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_C PROPERTIES PREFIX "")
set_target_properties(_C PROPERTIES SUFFIX "${LIBRARY_SUFFIX}")

target_compile_options(_C PUBLIC -fopenmp)
# add_link_options(_C PUBLIC -fopenmp)

target_include_directories(_C PUBLIC "." "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(_C CUDA::cudart)

install(TARGETS _C)
