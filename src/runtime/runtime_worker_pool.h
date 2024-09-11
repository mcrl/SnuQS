#ifndef _RUNTIME_WORKER_POOL_H_
#define _RUNTIME_WORKER_POOL_H_

#include <thread>
#include <vector>

namespace runtime {

class RuntimeQueue;
struct RuntimeWorkerPool {
  RuntimeWorkerPool(int nthreads, RuntimeQueue *_queue,
                    bool (*handler)(void *, void *), void *_arg);
  ~RuntimeWorkerPool();

  void *arg;

  void loop();
  RuntimeQueue *queue;
  bool (*handler)(void *entry, void *arg);
  std::vector<std::thread> threads;
  bool stop;
};

} // namespace runtime

#endif // _RUNTIME_WORKER_POOL_H_
