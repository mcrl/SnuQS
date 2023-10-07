add_library(quantum-decision-diagram
    STATIC 
    quantum-decision-diagram.cc
    )

#target_link_libraries(quantum-decision-diagram CUDA::cudart)
set_target_properties(quantum-decision-diagram PROPERTIES POSITION_INDEPENDENT_CODE ON)
