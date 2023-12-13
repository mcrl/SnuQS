add_library(transpile STATIC transpile.cc
                             )

target_link_libraries(transpile)
set_target_properties(transpile PROPERTIES POSITION_INDEPENDENT_CODE ON)
