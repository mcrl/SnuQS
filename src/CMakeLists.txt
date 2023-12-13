execute_process(COMMAND python3 -m pybind11 --includes
                OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

execute_process(COMMAND python3-config --extension-suffix
                OUTPUT_VARIABLE LIBRARY_SUFFIX)
string(STRIP "${LIBRARY_SUFFIX}" LIBRARY_SUFFIX)

add_library(_C SHARED module.cc)

include_directories("${PYBIND_INCLUDE_DIRS}" "." "./include")

set(CMAKE_CUDA_FLAGS
    ${CMAKE_CUDA_FLAGS}
    "--extended-lambda -fopenmp -I${CMAKE_CURRENT_LIST_DIR} -I${CMAKE_CURRENT_LIST_DIR}/include")
set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} 70)

set_target_properties(_C PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_C PROPERTIES PREFIX "")
set_target_properties(_C PROPERTIES SUFFIX "${LIBRARY_SUFFIX}")

target_link_libraries(
  _C
  PUBLIC optimizer
         buffer
         runtime
         circuit
         simulator
         #qasm_compiler
         transpile
         CUDA::cudart)

add_subdirectory(runtime)
add_subdirectory(circuit)
add_subdirectory(simulator)
add_subdirectory(transpile)
add_subdirectory(optimizer)
add_subdirectory(buffer)
