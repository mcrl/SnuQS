message(${CMAKE_CURRENT_SOURCE_DIR}/usr)
execute_process(COMMAND make prefix=${CMAKE_CURRENT_SOURCE_DIR}/usr install
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(libaio SHARED IMPORTED GLOBAL)
set_target_properties(libaio PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/usr/lib/libaio.so)
