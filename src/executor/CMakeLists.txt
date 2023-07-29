execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDE_DIRS)
string(REGEX REPLACE "-I" "" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(REGEX REPLACE " +" ";" PYBIND_INCLUDE_DIRS "${PYBIND_INCLUDE_DIRS}")
string(STRIP "${PYBIND_INCLUDE_DIRS}" PYBIND_INCLUDE_DIRS)

add_library(executor
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

  rt/rt_handle.cpp
  rt/rt_stream.cpp
  rt/rt_queue.cpp
  rt/rt_worker_pool.cpp
  rt/rt_task.cpp
  rt/rt_kernel.cpp
  rt/rt_host.cpp
  rt/rt_device.cpp
  rt/rt_storage.cpp
  rt/rt_soft_raid0.cpp
  )
target_include_directories(executor PUBLIC
  "${PYBIND_INCLUDE_DIRS}"
  )

#set_property(TARGET _executor PROPERTY POSITION_INDEPENDENT_CODE ON)
