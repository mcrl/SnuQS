add_library(compiler INTERFACE  "")

target_include_directories(compiler INTERFACE
  ${ANTLR_INCLUDE_DIRS}
  )

target_sources(compiler
  INTERFACE
  snuqasm-parser/snuqasmBaseListener.cpp
  snuqasm-parser/snuqasmLexer.cpp
  snuqasm-parser/snuqasmListener.cpp
  snuqasm-parser/snuqasmParser.cpp
  compiler.cpp
  preprocessor.cpp
  syntax_analyzer.cpp
  semantic_checker.cpp
  circuit_generator.cpp
  circuit_optimizer.cpp
  circuit_partitioner.cpp
  optimizer_utils.cpp
  gate_blocker.cpp
  initializer.cpp
  symbol_table.cpp
  permutation.cpp
  optimizer.cpp
  )

target_compile_options(compiler 
  INTERFACE 
  -Wno-attributes 
  -Wno-error=unused-variable)
target_compile_definitions(compiler 
  INTERFACE 
  _USE_BLOCKER_)

target_link_libraries(compiler INTERFACE
  circuit
  ${ANTLR_LIBRARIES}
)
