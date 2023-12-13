add_library(buffer STATIC memory_buffer.cc cuda_buffer.cc storage_buffer.cc
                          mt_raid.cc)

set_target_properties(buffer PROPERTIES POSITION_INDEPENDENT_CODE ON)
