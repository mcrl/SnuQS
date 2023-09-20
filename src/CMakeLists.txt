execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE LIBRARY_SUFFIX)
string(STRIP "${LIBRARY_SUFFIX}" LIBRARY_SUFFIX)

add_library(_snuqs
    SHARED 
    module.cpp
    )

target_include_directories(_snuqs PUBLIC
    "${PYBIND_INCLUDE_DIRS}"
    "."
    )

set_target_properties(_snuqs PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_snuqs PROPERTIES PREFIX "")
set_target_properties(_snuqs PROPERTIES SUFFIX "${LIBRARY_SUFFIX}")

target_link_libraries(_snuqs 
    PUBLIC
    executor 
    runtime

    quantum-circuit
    tensor-network
    dag

    optimizer
    CUDA::cudart
    )

add_subdirectory(executor)
add_subdirectory(runtime)

add_subdirectory(quantum-circuit)
add_subdirectory(tensor-network)
add_subdirectory(dag)
add_subdirectory(optimizer)
