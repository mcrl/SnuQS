#ifndef _STREAM_H_
#define _STREAM_H_

#include <memory>

class Event;
class Stream {
 public:
  Stream(void* stream);
  ~Stream();
  void* get();
  void synchronize();
  void wait_event(std::shared_ptr<Event> event);

  static std::shared_ptr<Stream> create();
  static std::shared_ptr<Stream> create_nonblocking();

 private:
  void* stream_;
};

#endif  //_STREAM_H_
