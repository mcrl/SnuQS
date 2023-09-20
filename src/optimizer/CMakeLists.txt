add_library(optimizer
    STATIC 
    optimizer.cc
    )

#target_link_libraries(optimizer CUDA::cudart)
set_target_properties(optimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
