#include "rt_stream.h"
#include "rt_device.h"
#include "rt_storage.h"
#include "rt_task.h"
#include "rt_kernel.h"

#include <algorithm>

namespace snuqs {
namespace rt {

static bool handler(void *entry, void *arg);

stream_handle_t::stream_handle_t(handle_t *_handle) 
  : handle(_handle) {
}

stream_handle_t::~stream_handle_t() {
}

RuntimeError stream_handle_t::create_stream(stream_t **streamp) {
  if (streamp == nullptr)
    return RT_NULL_POINTER;
  *streamp = new stream_t(this);
  return RT_SUCCESS;
}

RuntimeError stream_handle_t::destroy_stream(stream_t *stream) {
  if (stream == nullptr)
    return RT_NULL_POINTER;
  delete stream;
  return RT_SUCCESS;
}

stream_t::stream_t(stream_handle_t *_handle)
: handle(_handle) {
  this->pool = new worker_pool_t(1, &this->queue, 
      reinterpret_cast<bool (*)(void*, void*)>(&handler), this->handle->handle);
}

stream_t::~stream_t() {
  delete this->pool;
}

static bool handler(void *entry, void *arg) {
  handle_t *handle = reinterpret_cast<handle_t*>(arg);
  device_handle_t *device_handle = reinterpret_cast<device_handle_t*>(handle->device_handle);
  storage_handle_t *storage_handle = reinterpret_cast<storage_handle_t*>(handle->storage_handle);
  task_t* task = reinterpret_cast<task_t*>(entry);

  switch (task->status) {
    case task_t::status_t::ENQUEUED:
      task->set_status(task_t::status_t::RUNNING);
      switch (task->type) {
        case task_t::type_t::MEMCPY_H2D:
          device_handle->memcpy_h2d(task->dst, task->src, task->size);
          break;
        case task_t::type_t::MEMCPY_D2H:
          device_handle->memcpy_d2h(task->dst, task->src, task->size);
          break;
        case task_t::type_t::KERNEL:
          {
            kernel_t* kernel = reinterpret_cast<kernel_t*>(task);
            device_handle->launch_kernel(kernel->func, kernel->grid_dim, kernel->block_dim, kernel->args, kernel->shared_mem);
          }
          break;
        case task_t::type_t::MEMCPY_H2S:
          storage_handle->memcpy_h2s(task->dst, task->src, task->size);
          break;
        case task_t::type_t::MEMCPY_S2H:
          storage_handle->memcpy_s2h(task->dst, task->src, task->size);
          break;
      }
      task->set_status(task_t::status_t::DONE);
      return true;
    case task_t::status_t::RUNNING:
    case task_t::status_t::DONE:
    case task_t::status_t::CREATED:
      task->set_status(task_t::status_t::DONE);
      task->set_error(RT_ILLEGAL_TASK);
      return false;
  }
  return false;
}

RuntimeError stream_t::synchronize() {
  this->queue.wait_empty();
  return RuntimeError::RT_SUCCESS;
}

RuntimeError stream_t::enqueue_task(task_t *task) {
  if (task == nullptr) return RT_NULL_POINTER;

  task->set_status(task_t::status_t::ENQUEUED);
  this->queue.push_with_lock(task);
  this->queue.cv.notify_one();
  return RT_SUCCESS;
}

} // namespace rt 
} // namespace snuqs
