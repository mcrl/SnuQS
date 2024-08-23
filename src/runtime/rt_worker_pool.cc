#include "rt_worker_pool.h"
#include "rt_queue.h"
#include "rt_stream.h"


namespace snuqs {
namespace rt {

void worker_pool_t::loop() {
  while (!stop) {
    queue_t *queue = this->queue;
    std::unique_lock lk(queue->mutex);
    queue->cv.wait(lk, [queue, this]{
        return this->stop || queue->size() != 0;
        });

    if (queue->size() == 0 && this->stop) {
      lk.unlock();
      break;
    }

    void *entry = queue->front();
    task_t *task = reinterpret_cast<task_t*>(entry);
    bool done = this->handler(entry, this->arg);
    if (done) {
      queue->pop();
    }
    lk.unlock();
    if (done) {
      queue->cv.notify_one();
    }
  }
}

worker_pool_t::worker_pool_t(int nthreads, queue_t *_queue, bool (*_handler)(void*, void*), void *_arg)
  : queue(_queue), handler(_handler), arg(_arg)
{
  this->stop = false;
  for (int i = 0; i < nthreads; ++i) {
    this->threads.emplace_back(&worker_pool_t::loop, this);
  }
}

worker_pool_t::~worker_pool_t() {
  this->stop = true;
  this->queue->cv.notify_all();
  for (int i = 0; i < this->threads.size(); ++i) {
    this->threads[i].join();
  }
}

} // namespace rt
} // namespace snuqs
