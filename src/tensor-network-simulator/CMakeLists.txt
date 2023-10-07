add_library(tensor-network-simulator
    STATIC 
    tensor-network-simulator.cc
    )

set_target_properties(tensor-network-simulator PROPERTIES POSITION_INDEPENDENT_CODE ON)
