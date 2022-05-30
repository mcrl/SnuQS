add_library(transpiler "")


include_directories("../" "../../")
include_directories("/usr/local/cuda/include/")
target_sources(transpiler
	PRIVATE
	snuqlTranspiler.cpp
	transpiler.cpp
	)
