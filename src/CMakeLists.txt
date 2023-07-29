execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE LIBRARY_SUFFIX)
string(STRIP "${LIBRARY_SUFFIX}" LIBRARY_SUFFIX)

add_library(_snuqs
  SHARED 
  module.cpp

  executor/qop.cpp
  executor/qreg.cpp
  executor/creg.cpp
  executor/executor.cpp
  executor/simulator_omp.cpp 
  executor/simulator_cuda.cu 
  executor/simulator_multi_cuda.cu 
  executor/simulator_omp_io.cpp 
  executor/#simulator_multi_gpu.cu 

  executor/rt/rt_handle.cpp
  executor/rt/rt_stream.cpp
  executor/rt/rt_queue.cpp
  executor/rt/rt_worker_pool.cpp
  executor/rt/rt_task.cpp
  executor/rt/rt_kernel.cpp
  executor/rt/rt_host.cpp
  executor/rt/rt_device.cpp
  executor/rt/rt_storage.cpp
  executor/rt/rt_soft_raid0.cpp
  )

target_include_directories(_snuqs PUBLIC
  "${PYBIND_INCLUDE_DIRS}"
  )

set_target_properties(_snuqs PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_snuqs PROPERTIES PREFIX "")
set_target_properties(_snuqs PROPERTIES SUFFIX "${LIBRARY_SUFFIX}")

#target_link_libraries(_snuqs executor)

#add_subdirectory(executor)
