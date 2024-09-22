#include "rt_task.h"
#include <mutex>
#include <condition_variable>

namespace snuqs {
namespace rt {

// global lock for task synchronization
static std::mutex mutex;
static std::condition_variable cv;

task_t::task_t(type_t _type)
  : type(_type), status(status_t::CREATED), error(RT_SUCCESS) {
}

task_t::~task_t() {
}

void task_t::set_status(task_t::status_t status) {
  std::lock_guard<std::mutex> lock_guard(this->mutex);
  this->status = status;
}

void task_t::set_error(RuntimeError _error) {
  error = _error;
}

RuntimeError task_t::synchronize() {
  std::unique_lock lk(mutex);
  cv.wait(lk, [this]{
      return this->status == task_t::status_t::DONE;
      });

  lk.unlock();
  return this->error;
}

} // namespace rt
} // namespace snuqs
