#ifndef __QASM_PARASER_H__
#define __QASM_PARASER_H__

#include "antlr4-runtime.h"
#include <string>

namespace snuqs {
class QasmParser {
    antlr4::ParserRuleContext parse(const std::string &qasm);
};
}
#endif //__QASM_PARASER_H__
