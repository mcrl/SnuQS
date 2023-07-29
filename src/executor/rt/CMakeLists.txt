execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE LIBRARY_SUFFIX)
string(STRIP "${LIBRARY_SUFFIX}" LIBRARY_SUFFIX)

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(_rt
  SHARED 
  rt_handle.cpp
  rt_stream.cpp
  rt_queue.cpp
  rt_worker_pool.cpp
  rt_task.cpp
  rt_kernel.cpp
  rt_host.cpp
  rt_device.cpp
  rt_storage.cpp
  rt_soft_raid0.cpp
  #rt_event.cpp
  )
set_property(TARGET _rt PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(_rt PUBLIC
  "${PYBIND_INCLUDE_DIRS}"
  )
target_link_libraries(_rt
  CUDA::cudart)

set_target_properties(_rt PROPERTIES PREFIX "")
set_target_properties(_rt PROPERTIES SUFFIX "${LIBRARY_SUFFIX}")
