set(CMAKE_CUDA_ARCHITECTURES "70")


add_executable(snuqs 
	main.cpp
	job.cpp
	quantumCircuit.cpp
	gpu_utils.cpp
	tensor.cpp
	)

add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall>)
#add_compile_options($<$<COMPILE_LANGUAGE:C>:-Werror>)
add_compile_options($<$<COMPILE_LANGUAGE:C>:-O3>)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

set_target_properties(snuqs
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		)

include_directories("gate" "compiler" "transpiler" "simulator" "socl" "analyzer" "${BOOST_INCLUDE}" "${LIBURING_INCLUDE}")

add_subdirectory(gate)
add_subdirectory(compiler)
add_subdirectory(transpiler)
add_subdirectory(simulator)
add_subdirectory(socl)
add_subdirectory(analyzer)

include_directories("/usr/local/cuda/include")


target_link_libraries(snuqs gate compiler transpiler simulator socl analyzer 
"-L/usr/local/cuda/lib64" cudart
"-L/home2/daeyoung/libcutensor-linux-x86_64-1.5.0.3-archive/lib/11" cutensor
"-L/home2/daeyoung/cuquantum-linux-x86_64-22.05.0.41-archive/lib" cutensornet
boost_json
)
