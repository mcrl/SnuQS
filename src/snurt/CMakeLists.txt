add_library(snurt "")


target_sources(snurt
		PRIVATE
		api.cc
		context.cc
		dispatcher.cc
		command_queue.cc

		blkio.cc
		libio.cc
		
		api_io.cc
		context_io.cc
		worker_io.cc

		memory_allocator.cc

		api_cuda.cu
		context_cuda.cu
		worker_cuda.cu
		)

target_link_libraries(snurt
  PRIVATE
  libaio
  )
