#ifndef _RUNTIME_TASK_H_
#define _RUNTIME_TASK_H_

#include "runtime_error.h"
#include "runtime_types.h"
#include <mutex>

namespace runtime {

class RuntimeTask {
  enum class RuntimeTaskStatus {
    CREATED,
    ENQUEUED,
    RUNNING,
    DONE,
  };

  enum class RuntimeTaskType {
    EVENT,

    MEMCPY_H2D,
    MEMCPY_D2H,
    KERNEL,

    MEMCPY_H2S,
    MEMCPY_S2H,
  };

  RuntimeTask(RuntimeTaskType _type);
  virtual ~RuntimeTask();
  void set_status(RuntimeTask::RuntimeTaskStatus status);
  void set_error(RuntimeError error);
  RuntimeError synchronize();

  RuntimeTaskType type;
  RuntimeTaskStatus status;
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

} // namespace runtime

#endif // _RUNTIME_TASK_H_
