add_library(simulator STATIC statevector_simulator.cc
                             executor.cc
                             transpile.cc
                             run_single_gpu.cc
                             run_multi_gpu.cc
                             run_cpu.cc
                             run_storage.cc
                             qop_impl.cu
                             )

target_link_libraries(simulator buffer)
set_target_properties(simulator PROPERTIES POSITION_INDEPENDENT_CODE ON)
