#pragma once

#include <string>
#include <atomic>
#include "snuqasm-parser/snuqasmBaseListener.h"

namespace snuqs {

class Preprocessor : public snuqasmBaseListener {
	private:
	std::string base_dir_;
	std::string input_name_;
	std::string current_input_;
	std::string output_name_;
	std::atomic<unsigned int> count_;

	public:
	Preprocessor();
	void process(const std::string &filename);
	std::string ppFileName();

  std::string SearchFilePath(const std::string &filename);

	virtual void enterInclude(snuqasmParser::IncludeContext *ctx) override;
};

} // namespace snuqs
