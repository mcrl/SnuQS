#pragma once

#include <string>
#include "snuql-parser/snuqlBaseListener.h"

namespace snuqs {

class Preprocessor : public snuqlBaseListener {
	private:
	std::string input_name_;
	std::string output_name_;;

	public:
	Preprocessor();
	void process(const std::string &filename);
	std::string ppFileName();
	void enterHeader(snuqlParser::HeaderContext *ctx) override;
};

} // namespace snuqs
