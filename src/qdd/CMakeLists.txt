add_library(qdd
    STATIC 
    qdd.cc
    )

#target_link_libraries(qdd CUDA::cudart)
set_target_properties(qdd PROPERTIES POSITION_INDEPENDENT_CODE ON)
