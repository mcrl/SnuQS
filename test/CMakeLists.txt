include_directories(PUBLIC
  "${CMAKE_SOURCE_DIR}/src"
  )

add_executable(test)
target_sources(test PRIVATE main.cpp)
target_link_libraries(test 
  snuqs
  )
