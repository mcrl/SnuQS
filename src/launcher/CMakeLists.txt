add_library(launcher
    STATIC
    launcher.cc
    )

execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

target_include_directories(launcher PUBLIC
    "${PYBIND_INCLUDE_DIRS}"
    ".."
    )

set_target_properties(launcher PROPERTIES POSITION_INDEPENDENT_CODE ON)
