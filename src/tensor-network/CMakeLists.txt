add_library(tensor-network
    STATIC 
    tensor-network.cc
    )

#target_link_libraries(tensor-network CUDA::cudart)
set_target_properties(tensor-network PROPERTIES POSITION_INDEPENDENT_CODE ON)
