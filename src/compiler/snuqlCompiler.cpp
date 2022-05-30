#include "antlr4-runtime.h"
#include "snuql-parser/snuqlLexer.h"
#include "snuql-parser/snuqlParser.h"
#include "snuql-parser/snuqlBaseListener.h"

#include "snuqlCompiler.hpp"

#include "symbolTable.hpp"
#include "preprocessor.hpp"
#include "semanticChecker.hpp"
#include "circuitGenerator.hpp"
#include "semanticChecker.hpp"
#include <spdlog/spdlog.h>

#include <map>
#include <string>

using namespace antlr4;

namespace snuqs {

void SnuQLCompiler::compile(const std::string &filename) {
	// Preprocessor
	Preprocessor prep;
	prep.process(filename);
	std::string fn = prep.ppFileName();

	// Build Parse Tree
	std::ifstream stream(fn);
	ANTLRInputStream input(stream);
	snuqlLexer lexer(&input);
	CommonTokenStream tokens(&lexer);
	snuqlParser parser(&tokens);
	tree::ParseTree *tree = parser.mainprogram();

	// Semantic Check
	SemanticChecker sem_checker;
	sem_checker.check(tree);

	// Circuit Generator
	CircuitGenerator circ_gen(sem_checker.getSymbolTable());
	circ_ = circ_gen.generate(tree, std::cout);
}

const QuantumCircuit& SnuQLCompiler::getQuantumCircuit() {
	return circ_;
}

} // namespace snuqs
