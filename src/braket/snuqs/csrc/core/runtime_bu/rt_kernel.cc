#include "rt_kernel.h"
#include <cstring>
#include <cstdlib>

namespace snuqs {
namespace rt {

kernel_t::kernel_t()
  : task_t(task_t::type_t::KERNEL),
  args(nullptr), num_args(0) {
}

kernel_t::~kernel_t() {
  if (args != nullptr) {
    for (int i = 0; i < num_args; ++i) {
      if (args[i] != nullptr)
        std::free(args[i]);
    }
    std::free(args);
  }
}

RuntimeError kernel_t::set_arg(int idx, uint64_t size, void *ptr) {
  if (num_args <= idx) {
    args = reinterpret_cast<void**>(realloc(args, sizeof(void*) * (idx+1)));
    for (int i = num_args; i < idx+1; ++i) {
      args[i] = nullptr;
    }
    num_args = idx+1;
  }
  args[idx] = realloc(args[idx], size);
  memcpy(args[idx], ptr, size);
  return RT_SUCCESS;
}

} // namespace rt
} // namespace snuqs
