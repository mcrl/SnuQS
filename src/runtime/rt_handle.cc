#include "rt_handle.h"
#include "rt_host.h"
#include "rt_device.h"
#include "rt_storage.h"
#include "rt_stream.h"
#include "rt_task.h"
#include "rt_kernel.h"

namespace snuqs {
namespace rt {

handle_t::handle_t() {
  this->host_handle = new host_handle_t(this);
  this->device_handle = new device_handle_t(this);
  this->storage_handle = nullptr;
  this->stream_handle = new stream_handle_t(this);
}

handle_t::handle_t(std::vector<std::string> paths) {
  this->host_handle = new host_handle_t(this);
  this->device_handle = new device_handle_t(this);
  this->storage_handle = new storage_handle_t(this, paths);
  this->stream_handle = new stream_handle_t(this);
}

handle_t::~handle_t() {
  if (this->storage_handle)
    delete reinterpret_cast<storage_handle_t*>(this->storage_handle);
  delete reinterpret_cast<device_handle_t*>(this->device_handle);
  delete reinterpret_cast<stream_handle_t*>(this->stream_handle);
}

RuntimeError handle_t::create_stream(stream_t **stream_pp) {
  stream_handle_t *hndl = reinterpret_cast<stream_handle_t*>(this->stream_handle);
  return hndl->create_stream(stream_pp);
}

RuntimeError handle_t::destroy_stream(stream_t *stream) {
  stream_handle_t *hndl = reinterpret_cast<stream_handle_t*>(this->stream_handle);
  return hndl->destroy_stream(stream);
}

RuntimeError handle_t::stream_synchronize(stream_t *stream) {
  /* TODO: EVENT SYNC
  RuntimeError error;
  task_t *task = new task_t(task_t::type_t::EVENT);
  error = stream->enqueue_task(task);
  if (error != RT_SUCCESS)
    return error;

  return task->synchronize();
  */
  return stream->synchronize();
}

RuntimeError handle_t::malloc_host(addr_t *addr_p, uint64_t size) {
  host_handle_t *hndl = reinterpret_cast<host_handle_t*>(this->host_handle);
  return hndl->alloc(addr_p, size);
}

RuntimeError handle_t::malloc_storage(addr_t *addr_p, uint64_t size) {
  storage_handle_t *hndl = reinterpret_cast<storage_handle_t*>(this->storage_handle);
  return hndl->alloc(addr_p, size);
}

RuntimeError handle_t::malloc_device(addr_t *addr_p, uint64_t size) {
  device_handle_t *hndl = reinterpret_cast<device_handle_t*>(this->device_handle);
  return hndl->alloc(addr_p, size);
}

RuntimeError handle_t::malloc_pinned(addr_t *addr_p, uint64_t size) {
  device_handle_t *hndl = reinterpret_cast<device_handle_t*>(this->device_handle);
  return hndl->alloc_pinned(addr_p, size);
}

RuntimeError handle_t::free_host(addr_t addr) {
  host_handle_t *hndl = reinterpret_cast<host_handle_t*>(this->host_handle);
  return hndl->free(addr);
}

RuntimeError handle_t::free_storage(addr_t addr) {
  storage_handle_t *hndl = reinterpret_cast<storage_handle_t*>(this->storage_handle);
  return hndl->free(addr);
}

RuntimeError handle_t::free_device(addr_t addr) {
  device_handle_t *hndl = reinterpret_cast<device_handle_t*>(this->device_handle);
  return hndl->free(addr);
}

RuntimeError handle_t::free_pinned(addr_t addr) {
  device_handle_t *hndl = reinterpret_cast<device_handle_t*>(this->device_handle);
  return hndl->free_pinned(addr);
}

RuntimeError handle_t::get_num_devices(int *num_devices_p) {
  device_handle_t *hndl = reinterpret_cast<device_handle_t*>(this->device_handle);
  return hndl->get_num_devices(num_devices_p);
}

RuntimeError handle_t::get_device(int *device_p) {
  device_handle_t *hndl = reinterpret_cast<device_handle_t*>(this->device_handle);
  return hndl->get_device(device_p);
}

RuntimeError handle_t::set_device(int device) {
  device_handle_t *hndl = reinterpret_cast<device_handle_t*>(this->device_handle);
  return hndl->set_device(device);
}

RuntimeError handle_t::memcpy_h2d(addr_t dst, addr_t src, uint64_t size, stream_t *stream) {
  task_t *task = new task_t(task_t::type_t::MEMCPY_H2D);
  task->dst = dst;
  task->src = src;
  task->size = size;
  return stream->enqueue_task(task);
}

RuntimeError handle_t::memcpy_d2h(addr_t dst, addr_t src, uint64_t size, stream_t *stream) {
  task_t *task = new task_t(task_t::type_t::MEMCPY_D2H);
  task->dst = dst;
  task->src = src;
  task->size = size;
  return stream->enqueue_task(task);
}

RuntimeError handle_t::create_kernel(kernel_t **kernelp, const void *func) {
  kernel_t *kernel = new kernel_t();
  kernel->func = func;
  *kernelp = kernel;
  return RT_SUCCESS;
}

RuntimeError handle_t::set_kernel_arg(kernel_t *kernel, int idx, uint64_t size, void *ptr) {
  return kernel->set_arg(idx, size, ptr);
}

RuntimeError handle_t::launch_kernel(kernel_t *kernel, dim3 grid_dim, dim3 block_dim, uint64_t shared_mem, stream_t *stream) {
  kernel->grid_dim = grid_dim;
  kernel->block_dim = block_dim;
  kernel->shared_mem = shared_mem;
  return stream->enqueue_task(reinterpret_cast<task_t*>(kernel));
}

RuntimeError handle_t::memcpy_h2s(addr_t dst, addr_t src, uint64_t size, stream_t *stream) {
  task_t *task = new task_t(task_t::type_t::MEMCPY_H2S);
  task->dst = dst;
  task->src = src;
  task->size = size;
  return stream->enqueue_task(task);
}

RuntimeError handle_t::memcpy_s2h(addr_t dst, addr_t src, uint64_t size, stream_t *stream) {
  task_t *task = new task_t(task_t::type_t::MEMCPY_S2H);
  task->dst = dst;
  task->src = src;
  task->size = size;
  return stream->enqueue_task(task);
}

} // namespace rt
} // namespace snuqs
