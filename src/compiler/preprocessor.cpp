#include "preprocessor.h"

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <filesystem>

#include "antlr4-runtime.h"
#include "snuqasm-parser/snuqasmLexer.h"
#include "snuqasm-parser/snuqasmParser.h"
#include "snuqasm-parser/snuqasmBaseListener.h"

#include "logger.h"

namespace snuqs {

Preprocessor::Preprocessor()
: count_(0) {
}

void Preprocessor::process(const std::string &filename) {
	input_name_ = filename;
  auto p = filename.find_last_of("/\\");
  if (p == std::string::npos) {
    base_dir_ = ".";
  } else {
    base_dir_ = filename.substr(0, p);
  }


  current_input_ = "";
  output_name_ = filename;
  while (current_input_ != output_name_) {
    current_input_ = output_name_;
    std::ifstream stream(current_input_);
    antlr4::ANTLRInputStream input(stream);
    snuqasmLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    snuqasmParser parser(&tokens);
    antlr4::tree::ParseTree *tree = parser.mainprogram();
    antlr4::tree::ParseTreeWalker::DEFAULT.walk(this, tree);
  }
}

std::string Preprocessor::ppFileName() {
	return output_name_;
}


std::string Preprocessor::SearchFilePath(const std::string &filename) {
  std::string candidate = base_dir_ + "/" + filename;
  std::ifstream f(candidate);
  if (f.good()) {
    return candidate;
  }

  return "";
}

void Preprocessor::enterInclude(snuqasmParser::IncludeContext *ctx)  {
  std::ifstream ifs(current_input_);
	std::vector<std::vector<std::string>> lines;
  std::string str;
	size_t nlines = 0;
	while (getline(ifs, str)) {
		lines.push_back({});
		lines[nlines].push_back(str);
		nlines++;
	}

	size_t line = ctx->getStart()->getLine() - 1;
	size_t col_start = ctx->getStart()->getCharPositionInLine();
	size_t col_end = ctx->getStop()->getCharPositionInLine();

	std::string the_line = lines[line][0];

	lines[line][0] = the_line.substr(0, col_start);
  {
    auto text = ctx->StringLiteral()->getText();
    auto filename = text.substr(1, text.length()-2);
    auto fn = SearchFilePath(filename);
    std::ifstream ifs(fn);
    if (!ifs.good()) {
      Logger::error("Cannot include \"{}\": No such file exists.\n", fn);
      std::exit(EXIT_FAILURE);
    }

    lines[line].push_back("// " + ctx->getText());
    while (getline(ifs, str)) {
      lines[line].push_back(str);
    }

  }
  lines[line].push_back(the_line.substr(col_end+1));


  std::size_t found = current_input_.find_last_of("/\\");
	output_name_ = "/tmp/" + current_input_.substr(found+1) + std::to_string(count_++);
	std::ofstream ofs(output_name_);
	for (auto & v : lines) {
		for (auto & l : v) {
		  ofs << l << "\n";
		}
	}
}

}
// namespace snuqs
