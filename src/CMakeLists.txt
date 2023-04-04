set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

include_directories(driver PUBLIC
  "${CMAKE_SOURCE_DIR}/src"
  )

add_library(snuqs SHARED "")

target_link_libraries(snuqs 
  simulator 
  compiler 
  circuit
  #snurt
  )

add_subdirectory(circuit)
add_subdirectory(compiler)
add_subdirectory(simulator)
