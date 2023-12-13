add_library(circuit
    STATIC 
    circuit.cc
    reg.cc
    arg.cc
    qop.cc
    parameter.cc
    )

set_target_properties(circuit PROPERTIES POSITION_INDEPENDENT_CODE ON)
