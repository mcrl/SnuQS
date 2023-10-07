add_library(statevector-simulator
    STATIC 
    statevector-simulator.cc
    )

#target_link_libraries(simulator CUDA::cudart)
set_target_properties(statevector-simulator PROPERTIES POSITION_INDEPENDENT_CODE ON)
