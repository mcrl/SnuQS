add_library(directed-acyclic-graph
    STATIC 
    directed-acyclic-graph.cc
    )

#target_link_libraries(directed-acyclic-graph CUDA::cudart)
set_target_properties(directed-acyclic-graph PROPERTIES POSITION_INDEPENDENT_CODE ON)
