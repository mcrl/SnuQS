add_library(quantum-decision-diagram-simulator
    STATIC 
    quantum-decision-diagram-simulator.cc
    )

set_target_properties(quantum-decision-diagram-simulator PROPERTIES POSITION_INDEPENDENT_CODE ON)
