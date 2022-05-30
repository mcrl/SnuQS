#include "antlr4-runtime.h"
#include "snuql-parser/snuqlLexer.h"
#include "snuql-parser/snuqlParser.h"
#include "snuql-parser/snuqlBaseListener.h"

#include "preprocessor.hpp"

#include <cstdio>
#include <cstddef>
#include <boost/filesystem.hpp>

namespace snuqs {

Preprocessor::Preprocessor() {
}

void Preprocessor::process(const std::string &filename) {
	input_name_ = filename;

	std::ifstream stream(filename);
	antlr4::ANTLRInputStream input(stream);
	snuqlLexer lexer(&input);
	antlr4::CommonTokenStream tokens(&lexer);
	snuqlParser parser(&tokens);
	antlr4::tree::ParseTree *tree = parser.mainprogram();
	antlr4::tree::ParseTreeWalker::DEFAULT.walk(this, tree);
}

std::string Preprocessor::ppFileName() {
	return output_name_;
}

void Preprocessor::enterHeader(snuqlParser::HeaderContext *ctx)  {
	if (ctx->include().size() == 0) {
		output_name_ = input_name_;
		return;
	}

	std::vector<std::string> lines;
	std::ifstream stream(input_name_);
	std::string line;
	while (getline(stream, line)) {
		lines.push_back(line);
	}

	std::vector<std::vector<std::string>> newlines(lines.size());
	for (size_t i = 0; i < lines.size(); ++i) {
		newlines[i].push_back(lines[i]);
	}

	snuqlParser::IncludeContext *first_inc = ctx->include()[0];
	snuqlParser::IncludeContext *last_inc = ctx->include()[ctx->include().size()-1];

	size_t start_line = first_inc->getStart()->getLine() - 1;
	size_t start_col = first_inc->getStart()->getCharPositionInLine();
	size_t stop_line = last_inc->getStop()->getLine() - 1;
	size_t stop_col = last_inc->getStop()->getCharPositionInLine();

	std::string sl = newlines[start_line][0].substr(0, start_col);
	std::string el = newlines[stop_line][0].substr(stop_col+1);
	for (size_t i = start_line; i <= stop_line; ++i) {
		newlines[i].erase(newlines[i].begin());
	}
	newlines[start_line].push_back(sl);
	newlines[stop_line].push_back(el);

	for (auto &&inc : ctx->include()) {
		auto text = inc->StringLiteral()->getText();
		auto fn = text.substr(1, text.length()-2);
		std::ifstream ifs(fn);
		while (getline(ifs, line)) {
			newlines[start_line].push_back(line);
		}
	}

	//output_name_ = std::mkstemp(nullptr);
	std::FILE *tmpf = std::tmpfile();

	// Linux-specific method to display the tmpfile name
	output_name_ = boost::filesystem::read_symlink(boost::filesystem::path("/proc/self/fd") / std::to_string(fileno(tmpf))).string();
	std::ofstream ofs(output_name_);
	for (auto && v : newlines) {
		for (auto && l : v) {
			ofs << l << "\n";
		}
	}
}

}
// namespace snuqs
