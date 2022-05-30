add_library(analyzer "")

include_directories("../../" ".." "../gate/" ${ANTLR4_INCLUDE} ${SPDLOG_INCLUDE})
include_directories("/usr/local/cuda/include/")
include_directories("${BOOST_INCLUDE}")

target_sources(analyzer
	PRIVATE
	circuitAnalyzer.cpp
	)

