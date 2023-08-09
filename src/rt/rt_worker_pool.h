#ifndef __RT_WORKER_POOL_H__
#define __RT_WORKER_POOL_H__

#include <vector>
#include <thread>

namespace snuqs {
namespace rt {

struct queue_t;
struct worker_pool_t {
  worker_pool_t(int nthreads, queue_t *_queue, bool (*handler)(void*, void*), void *_arg);
  ~worker_pool_t();

  void *arg;

  void loop();
  queue_t *queue;
  bool (*handler)(void *entry, void *arg);
  std::vector<std::thread> threads;
  bool stop;
};

} // namespace rt
} // namespace snuqs

#endif // __RT_WORKER_POOL_H__
