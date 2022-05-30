add_library(socl "")

include_directories("../" "/usr/local/cuda/include" "${LIBURING_INCLUDE}")

target_sources(socl
		PRIVATE
		socl.c
		baio.c
		socl-gpu.c
		socl-io.c
		)
target_compile_options(socl PRIVATE $<$<COMPILE_LANGUAGE:C>:-fopenmp -O3>)
target_link_libraries(socl "-L../../dep/liburing/src/" uring gomp aio)


add_executable(socl-test "")
target_sources(socl-test
		PRIVATE
		socl-test.c
		socl.c
		socl-gpu.c
		socl-io.c
		)

target_link_libraries(socl-test "-L${LIBURING_LIBRARY_PATH}" uring aio)
target_link_libraries(socl-test "-L/usr/local/cuda/lib64" cudart)

target_compile_options(socl-test PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall>)
target_compile_options(socl-test PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wno-error>)
target_compile_options(socl-test PRIVATE $<$<COMPILE_LANGUAGE:C>:-O3>)

#add_subdirectory(queues)
set_target_properties(socl-test
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		)

add_executable(socl-io-test "")
target_sources(socl-io-test
		PRIVATE
		socl-io-test.c
		socl.c
		socl-gpu.c
		socl-io.c
		)

target_link_libraries(socl-io-test "-L${LIBURING_LIBRARY_PATH}" uring aio)
target_link_libraries(socl-io-test "-L/usr/local/cuda/lib64" cudart)

target_compile_options(socl-io-test PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall>)
target_compile_options(socl-io-test PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wno-error>)
target_compile_options(socl-io-test PRIVATE $<$<COMPILE_LANGUAGE:C>:-O3>)

#add_subdirectory(queues)
set_target_properties(socl-io-test
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		)
