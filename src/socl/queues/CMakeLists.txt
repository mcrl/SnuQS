add_executable(test)
target_sources(test
		PRIVATE
		main.c
		spsdb.c
		spsdu.c
		)
set_target_properties(test
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		)
