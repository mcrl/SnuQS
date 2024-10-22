#ifndef __RT_EVENT_H__
#define __RT_EVENT_H__

#include "runtime_context.h"
#include "runtime_error.h"
#include <condition_variable>
#include <mutex>

namespace runtime {

class RuntimeEvent {
public:
  enum class RuntimeEventStatus {
    CREATED,
    DONE,
  };

  RuntimeEvent(RuntimeContext *ctx);

  RuntimeEventStatus status = RuntimeEventStatus::CREATED;
  RuntimeContext *hndl;
  std::mutex mutex;
  std::condition_variable cv;
};

static RuntimeError EventCreate(RuntimeContext *ctx, RuntimeEvent **event);
static RuntimeError EventDestroy(RuntimeContext *ctx, RuntimeEvent *event);

} // namespace runtime

#endif //__RT_EVENT_H__
