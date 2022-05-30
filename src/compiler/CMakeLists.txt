add_library(compiler "")

include_directories("../../" ".." "../gate/" ${ANTLR4_INCLUDE} ${SPDLOG_INCLUDE})
include_directories("/usr/local/cuda/include/")
include_directories("${BOOST_INCLUDE}")

target_sources(compiler
	PRIVATE
	snuql-parser/snuqlBaseListener.cpp
	snuql-parser/snuqlLexer.cpp
	snuql-parser/snuqlListener.cpp
	snuql-parser/snuqlParser.cpp
	circuitGenerator.cpp
	circuitOptimizer.cpp
	circuitPartitioner.cpp
	#OptimalPartitioner.cpp
	optimizerUtils.cpp
	gateBlocker.cpp
	initializer.cpp
	preprocessor.cpp
	semanticChecker.cpp
	symbolTable.cpp
	permutation.cpp
	snuqlCompiler.cpp
	snuqlOptimizer.cpp
	)

target_compile_options(compiler PRIVATE -Wno-attributes)
target_compile_definitions(compiler PRIVATE _USE_BLOCKER_)

target_link_libraries(compiler "-L../../dep/boost/stage/lib/ -L../../dep/antlr4/runtime/Cpp/run/usr/local/lib/" antlr4-runtime boost_filesystem boost_system)
