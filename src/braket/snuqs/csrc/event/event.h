#ifndef _EVENT_H_
#define _EVENT_H_

#include <memory>

#include "stream/stream.h"

class Event {
 public:
  Event(void* event);
  ~Event();
  void* get();
  void record(std::shared_ptr<Stream> stream);
  void synchronize();

  static std::shared_ptr<Event> create();

 private:
  void* event_;
};

#endif  //_EVENT_H_
