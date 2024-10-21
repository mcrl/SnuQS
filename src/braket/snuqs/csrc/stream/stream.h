#ifndef _STREAM_H_
#define _STREAM_H_

#include <memory>

class Stream {
 public:
  Stream(void* stream);
  ~Stream();
  void* get();

  static std::shared_ptr<Stream> create();

 private:
  void* stream_;
};

#endif  //_STREAM_H_
