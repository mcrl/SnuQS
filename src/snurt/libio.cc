#include "libio.h"

#include <stdexcept>

#include "logger.h"

namespace snurt {
namespace blkio {

IOContext::IOContext() {

  memset(&__ctx, 0, sizeof(io_context_t));
  int err;
  err = io_setup(32, &__ctx);
  if (err) {
    throw std::runtime_error("io_setup failed");
  }
}

IOContext::IOContext(unsigned nr_events) {

  memset(&__ctx, 0, sizeof(io_context_t));
  int err;
  err = io_setup(nr_events, &__ctx);
  if (err) {
    throw std::runtime_error("io_setup failed");
  }
}

IOContext::~IOContext () {
  int err;
  err = io_destroy(__ctx);
  if (err) {
    throw std::runtime_error("io_destroy failed");
  }
}

IOContext::IOContext(const IOContext &other) {
  if (this != &other) {
    __ctx = other.__ctx;
  }
}

int IOEvent::Result() const {
  return __io_event.res;
}


int IOSetup(unsigned nr_events, IOContext *ctx) {
  return io_setup(nr_events, &ctx->__ctx);
}

void IOPrepPwrite(IOControlBlock *cb, int fd, void *buf, size_t count, long long offset) {
  io_prep_pwrite(&cb->__iocb, fd, buf, count, offset);
}

void IOPrepPread(IOControlBlock *cb, int fd, void *buf, size_t count, long long offset) {
  io_prep_pread(&cb->__iocb, fd, buf, count, offset);
}

int IOSubmitOne(const IOContext &ctx, IOControlBlock *cbp) {
  iocb *iocbp = &cbp->__iocb;
  return io_submit(ctx.__ctx, 1, &iocbp);
}

int IOGetEventsOne(const IOContext &ctx, IOEvent *event, TimeSpec *ts) {
  return io_getevents(ctx.__ctx, 1, 1, &event->__io_event, (ts == nullptr) ? nullptr : &ts->timespec);
}

} // namespace blkio
} // namespace snurt
