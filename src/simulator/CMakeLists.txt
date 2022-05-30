add_library(simulator "")

include_directories("../")
include_directories("/usr/local/cuda/include")
include_directories(${CUTENSORNET_ROOT}/include)
include_directories("/home2/daeyoung/cuquantum-linux-x86_64-22.05.0.41-archive/include")


target_sources(simulator 
	PRIVATE
	simulator.cpp
	statevector_cpu.cpp
	statevector_cpu_io.cpp
	statevector_gpu.cpp
	statevector_gpu_io.cpp
	density_cpu.cpp
	density_cpu_io.cpp
	density_gpu.cpp
	density_gpu_io.cpp
	contraction_cpu.cpp
	contraction_gpu.cpp
	)

include_directories("/usr/local/cuda/include")
