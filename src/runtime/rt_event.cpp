#include "rt_event.h"
#include "rt_stream.h"

#include <mutex>
#include <condition_variable>

namespace snuqs {
namespace rt {

enum class EventStatus {
  CREATED,
  DONE,
};

struct _event_t {
  EventStatus status = EventStatus::CREATED;
  handle_t *hndl;
  std::mutex mutex;
  std::condition_variable cv;
};

event_t create_event(handle_t handle) {
  _event_t *evt = new _event_t();
  evt->hndl = &handle;

  //register_event(evt->hndl, evt);

  event_t event;
  event.obj = evt;
  return event;
}

void destroy_event(handle_t handle, event_t event) {
  _event_t *evt = reinterpret_cast<_event_t*>(event.obj);
  //register_event(evt->hndl, evt);
  delete reinterpret_cast<_event_t*>(event.obj);
}

RuntimeError event_record(handle_t handle, event_t event, stream_t stream) {
  _event_t *evt = reinterpret_cast<_event_t*>(event.obj);
  return stream_enqueue(stream, TaskType::EVENT, (std::size_t) evt, 0);
}

RuntimeError event_synchronize(handle_t handle, event_t event) {
  _event_t *evt = reinterpret_cast<_event_t*>(event.obj);
  std::unique_lock lk(evt->mutex);
  evt->cv.wait(lk, [evt]{
      return evt->status == EventStatus::DONE;
      });
  lk.unlock();
  evt->cv.notify_all();
  return RuntimeError::RT_SUCCESS;
}

RuntimeError stream_wait_event(handle_t handle, stream_t stream, event_t event) {
  _event_t *evt = reinterpret_cast<_event_t*>(event.obj);

  return stream_enqueue(stream, TaskType::WAIT_EVENT, (std::size_t) evt, 0);
}

} // namespace rt
} // namespace snuqs
