

#include "qasm_compiler/qasm_compiler.h"
#include "qasm_compiler/qasm_parser.h"
#include "qasm_compiler/qasm_preprocessor.h"

namespace snuqs {

std::shared_ptr<Circuit> QasmCompiler::compile(std::string file_name) {
  std::shared_ptr<Circuit> circ = std::make_shared<Circuit>(file_name);

  QasmPreprocessor pp;
  std::string preprocessed_qasm = pp.preprocess(file_name);

  QasmParser parser;
  auto tree = parser.parse(qasm_preprocessed);
  return circ;
}

} // namespace snuqs
