#ifndef _RUNTIME_QUEUE_H_
#define _RUNTIME_QUEUE_H_

#include "runtime_types.h"
#include <condition_variable>
#include <deque>
#include <mutex>

namespace runtime {

struct RuntimeQueue {
  using reference_t = std::deque<void *>::iterator;
  using const_reference_t = std::deque<void *>::const_iterator;

  void push(void *task);
  void push_with_lock(void *task);
  void *front() const;
  void pop();
  uint64_t size() const;
  reference_t begin();
  reference_t end();
  const_reference_t cbegin() const;
  const_reference_t cend() const;
  void wait_empty();

  std::deque<void *> queue;
  std::mutex mutex;
  std::condition_variable cv;
};

} // namespace runtime

#endif // _RUNTIME_QUEUE_H_
