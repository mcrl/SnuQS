add_library(quantum-circuit
    STATIC 
    quantum-circuit.cc
    quantum-operation.cc
    quantum-register.cc
    )

#target_link_libraries(quantum-circuit CUDA::cudart)
set_target_properties(quantum-circuit PROPERTIES POSITION_INDEPENDENT_CODE ON)
