execute_process(COMMAND python3 -m pybind11 --includes
                OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

execute_process(COMMAND python3-config --extension-suffix
                OUTPUT_VARIABLE LIBRARY_SUFFIX)
string(STRIP "${LIBRARY_SUFFIX}" LIBRARY_SUFFIX)

add_library(_snuqs_impl SHARED module.cc)

include_directories("${PYBIND_INCLUDE_DIRS}" "." "./include")

set_target_properties(_snuqs_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_snuqs_impl PROPERTIES PREFIX "")
set_target_properties(_snuqs_impl PROPERTIES SUFFIX "${LIBRARY_SUFFIX}")

target_link_libraries(
  _snuqs_impl
  PUBLIC launcher
         optimizer
         buffer
         runtime
         quantum_circuit
         dag
         simulator
         CUDA::cudart)

add_subdirectory(launcher)
add_subdirectory(runtime)

add_subdirectory(quantum_circuit)
add_subdirectory(dag)

add_subdirectory(simulator)
add_subdirectory(optimizer)
add_subdirectory(buffer)