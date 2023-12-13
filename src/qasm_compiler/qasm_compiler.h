#ifndef __QASM_COMPILER_H__
#define __QASM_COMPILER_H__
#include "circuit/circuit.h"

namespace snuqs {
class QasmCompiler {
  std::shared_ptr<Circuit> compile(std::string file_name);
};
} // namespace snuqs

#endif //__QASM_COMPILER_H__
