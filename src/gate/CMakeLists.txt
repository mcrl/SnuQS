add_library(gate "")

include_directories("../")
include_directories("/usr/local/cuda/include")

#target_compile_options(gate PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_ARCH} --extended-lambda --std=c++17 -O3>)
target_sources(gate
	PRIVATE
	gate.cpp 
	gate_cpu.cpp 
	gate_gpu.cu
	)

target_include_directories(snuqs PUBLIC
	"${CMAKE_CURRENT_LIST_DIR}/.."
	"/usr/local/cuda/include"
)
link_directories("/usr/local/cuda/lib64/")
