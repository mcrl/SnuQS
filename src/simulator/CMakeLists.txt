add_library(simulator INTERFACE "")

target_sources(simulator 
  INTERFACE
  job.cpp 
	simulator.cpp
	nosimulator.cpp
	statevector_cpu.cpp
	statevector_cpu_io.cpp

  #statevector_gpu.cpp
  #statevector_gpu_io.cpp
	)

