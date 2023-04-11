execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE LIBRARY_SUFFIX)
string(STRIP "${LIBRARY_SUFFIX}" LIBRARY_SUFFIX)
add_library(impl 
  SHARED 
  compiler.cpp
  _snuqs.cpp
  )

target_include_directories(impl PUBLIC
  "${CMAKE_SOURCE_DIR}/src"
  "${PYBIND_INCLUDE_DIRS}"
  )

set_target_properties(impl PROPERTIES PREFIX "")
set_target_properties(impl PROPERTIES SUFFIX "${LIBRARY_SUFFIX}")
target_link_libraries(impl 
  snuqs
  )

