#ifndef __RT_TASK_H__
#define __RT_TASK_H__

#include "rt_type.h"
#include "rt_error.h"
#include <mutex>

namespace snuqs {
namespace rt {

struct task_t {
  enum class status_t {
    CREATED,
    ENQUEUED,
    RUNNING,
    DONE,
  };

  enum class type_t {
    EVENT,

    MEMCPY_H2D,
    MEMCPY_D2H,
    KERNEL,

    MEMCPY_H2S,
    MEMCPY_S2H,
  };

  task_t(type_t _type);
  virtual ~task_t();
  void set_status(task_t::status_t status);
  void set_error(RuntimeError error);
  RuntimeError synchronize();

  type_t type;
  status_t status;
  RuntimeError error;

  std::mutex mutex;

  union {
    addr_t dst;
    const void *func;
  };

  union {
    addr_t src;
    void **args;
  };

  union {
    uint64_t size;
    uint64_t shared_mem;
  };


};

} // namespace rt
} // namespace snuqs

#endif // __RT_TASK_H__
