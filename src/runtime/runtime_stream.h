#ifndef _RT_STREAM_H_
#define _RT_STREAM_H_

#include "runtime_context.h"
#include "runtime_error.h"
#include "runtime_queue.h"
#include "runtime_task.h"
#include "runtime_worker_pool.h"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace runtime {

struct RuntimeStream {
private:
  RuntimeStream(RuntimeContext *ctx);
  ~RuntimeStream();

public:
  RuntimeContext *handle;

  RuntimeQueue queue;
  RuntimeWorkerPool *pool;

  RuntimeError synchronize();
  RuntimeError enqueueTAsk(RuntimeTask *task);
};

static RuntimeError StreamCreate(RuntimeContext *ctx, RuntimeStream **streamp);
static RuntimeError StreamDestroy(RuntimeContext *ctx, RuntimeStream *stream);

} // namespace runtime

#endif // _RT_STREAM_H_
