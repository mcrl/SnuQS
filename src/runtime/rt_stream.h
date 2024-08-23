#ifndef __RT_STREAM_H__
#define __RT_STREAM_H__

#include "rt_handle.h"
#include "rt_error.h"
#include "rt_task.h"

#include "rt_queue.h"
#include "rt_worker_pool.h"

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace snuqs {
namespace rt {

struct event_t;

struct stream_t;
struct stream_handle_t {
  stream_handle_t(handle_t *_handle);
  ~stream_handle_t();

  handle_t *handle;
  RuntimeError create_stream(stream_t **streamp);
  RuntimeError destroy_stream(stream_t *stream);
};

struct stream_t {
  friend stream_handle_t;
  private:
  stream_t(stream_handle_t *_handle);
  ~stream_t();

  public:

  int hh = 5;
  stream_handle_t *handle;

  queue_t queue;
  worker_pool_t *pool;

  RuntimeError synchronize();
  RuntimeError enqueue_task(task_t *task);
};


} // namespace snuqs
} // namespace rt 

#endif // __RT_STREAM_H__
