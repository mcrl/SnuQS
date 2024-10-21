#ifndef __RT_KERNEL_H__
#define __RT_KERNEL_H__

#include "rt_task.h"
#include "rt_type.h"
#include "rt_error.h"

namespace snuqs {
namespace rt {

struct kernel_t : public task_t {
  kernel_t();
  ~kernel_t();

  RuntimeError set_arg(int idx, uint64_t size, void *ptr);

  dim3 grid_dim;
  dim3 block_dim;

  void **args;
  int num_args; 
};

} // namespace rt
} // namespace snuqs

#endif // __RT_KERNEL_H__
