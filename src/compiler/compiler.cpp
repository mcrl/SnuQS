#include "compiler.h"

#include <map>
#include <string>


#include "antlr4-runtime.h"

#include "snuqasm-parser/snuqasmLexer.h"
#include "snuqasm-parser/snuqasmParser.h"
#include "snuqasm-parser/snuqasmBaseListener.h"
#include "symbol_table.h"
#include "semantic_checker.h"
#include "circuit_generator.h"

#include "preprocessor.h"
#include "syntax_analyzer.h"
#include "semantic_checker.h"

#include "logger.h"

using namespace antlr4;

namespace snuqs {

void Compiler::Compile(const std::string &filename) {
	std::ifstream ifs(filename);
	if (!ifs.good()) {
	  Logger::error("No such file exists {}\n", filename);
	  std::exit(EXIT_FAILURE);
  }

  SyntaxAnalyzer sa;
  sa.analyze(filename);

	// Preprocessor
	Preprocessor pp;
	pp.process(filename);
	std::string fn = pp.ppFileName();

	// Build Parse Tree
	std::ifstream stream(fn);
	ANTLRInputStream input(stream);
	snuqasmLexer lexer(&input);
	CommonTokenStream tokens(&lexer);
	snuqasmParser parser(&tokens);
	tree::ParseTree *tree = parser.mainprogram();

	// Semantic Check
	SemanticChecker sem_checker;
	sem_checker.check(tree);

	// Circuit Generator
	CircuitGenerator circ_gen(sem_checker.getSymbolTable());
	circ_ = circ_gen.Generate(tree);
}

const QuantumCircuit& Compiler::GetQuantumCircuit() {
	return circ_;
}

} // namespace snuqs
