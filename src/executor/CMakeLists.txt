add_library(_executor
    SHARED 
    qop.cpp
    qreg.cpp
    creg.cpp
    executor.cpp
    simulator_omp.cpp 
    simulator_cuda.cu 
    simulator_multi_cuda.cu 
    simulator_omp_io.cpp 
    #simulator_multi_gpu.cu 
    )

execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

target_include_directories(_executor PUBLIC
    "${PYBIND_INCLUDE_DIRS}"
    ".."
    )

set_target_properties(_executor PROPERTIES POSITION_INDEPENDENT_CODE ON)
