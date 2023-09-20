add_library(dag
    STATIC 
    dag.cc
    )

#target_link_libraries(dag CUDA::cudart)
set_target_properties(dag PROPERTIES POSITION_INDEPENDENT_CODE ON)
