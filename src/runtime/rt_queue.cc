#include "rt_queue.h"

namespace snuqs {
namespace rt {

void queue_t::push(void *task) {
  this->queue.push_back(task);
}

void queue_t::push_with_lock(void *task) {
  std::lock_guard<std::mutex> lock_guard(this->mutex);
  this->push(task);
}

void* queue_t::front() const {
  return this->queue.front();
}

void queue_t::pop() {
  this->queue.pop_front();
}

uint64_t queue_t::size() const {
  return this->queue.size();
}

queue_t::reference_t queue_t::begin() {
  return this->queue.begin();
}

queue_t::reference_t queue_t::end() {
  return this->queue.end();
}

queue_t::const_reference_t queue_t::cbegin() const {
  return this->queue.cbegin();
}

queue_t::const_reference_t queue_t::cend() const {
  return this->queue.cend();
}

void queue_t::wait_empty() {
  std::unique_lock lk(this->mutex);
  this->cv.wait(lk, [this]{
      return this->queue.size() == 0;
      });
  lk.unlock();
}

} // namespace rt
} // namespace snuqs
