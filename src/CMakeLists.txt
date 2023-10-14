execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE LIBRARY_SUFFIX)
string(STRIP "${LIBRARY_SUFFIX}" LIBRARY_SUFFIX)

add_library(_snuqs_impl
    SHARED 
    module.cpp
    )

include_directories(
    "${PYBIND_INCLUDE_DIRS}"
    "."
    )

set_target_properties(_snuqs_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_snuqs_impl PROPERTIES PREFIX "")
set_target_properties(_snuqs_impl PROPERTIES SUFFIX "${LIBRARY_SUFFIX}")

target_link_libraries(_snuqs_impl 
    PUBLIC
    launcher 
    runtime

    quantum-circuit
    tensor-network
    quantum-decision-diagram
    directed-acyclic-graph


    statevector-simulator
    tensor-network-simulator
    quantum-decision-diagram-simulator

    optimizer

    #misc

    CUDA::cudart
    )

add_subdirectory(launcher)
add_subdirectory(runtime)

add_subdirectory(quantum-circuit)
add_subdirectory(tensor-network)
add_subdirectory(directed-acyclic-graph)
add_subdirectory(quantum-decision-diagram)

add_subdirectory(statevector-simulator)
add_subdirectory(tensor-network-simulator)
add_subdirectory(quantum-decision-diagram-simulator)

add_subdirectory(optimizer)
