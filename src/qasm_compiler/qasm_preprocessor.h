#ifndef __QASM_PREPROCESSOR_H__
#define __QASM_PREPROCESSOR_H__

#include <string>

namespace snuqs {
class QasmPreprocessor {
    public:
  std::string preprocess(const std::string &file_name) const;
};
} // namespace snuqs

#endif // __QASM_PREPROCESSOR_H__
