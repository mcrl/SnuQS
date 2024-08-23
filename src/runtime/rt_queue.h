#ifndef __RT_QUEUE_H__
#define __RT_QUEUE_H__

#include "rt_type.h"
#include <deque>
#include <mutex>
#include <condition_variable>

namespace snuqs {
namespace rt {

struct queue_t {
  using reference_t = std::deque<void*>::iterator;
  using const_reference_t = std::deque<void*>::const_iterator;

  void push(void *task);
  void push_with_lock(void *task);
  void* front() const;
  void pop();
  uint64_t size() const;
  reference_t begin();
  reference_t end();
  const_reference_t cbegin() const;
  const_reference_t cend() const;
  void wait_empty();

  std::deque<void*> queue;
  std::mutex mutex;
  std::condition_variable cv;
};

} // namespace rt
} // namespace snuqs

#endif // __RT_QUEUE_H__
