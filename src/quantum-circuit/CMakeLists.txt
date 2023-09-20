add_library(quantum-circuit
    STATIC 
    quantum-circuit.cc
    )

#target_link_libraries(quantum-circuit CUDA::cudart)
set_target_properties(quantum-circuit PROPERTIES POSITION_INDEPENDENT_CODE ON)
