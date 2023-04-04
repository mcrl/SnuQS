#pragma once

extern "C" {
#include <libaio.h>
}

namespace snurt {
namespace blkio {

struct IOContext {
  IOContext();
  IOContext(unsigned nr_events);
  IOContext(const IOContext &other);
  ~IOContext ();
  io_context_t __ctx;
};

struct IOControlBlock {
  iocb __iocb;
};

struct IOEvent {
  io_event __io_event;
  int Result() const;
};

struct TimeSpec {
  struct timespec timespec;
};

int IOSetup(unsigned nr_events, IOContext *ctx);
void IOPrepPwrite(IOControlBlock *cb, int fd, void *buf, size_t count, long long offset);
void IOPrepPread(IOControlBlock *cb, int fd, void *buf, size_t count, long long offset);
int IOSubmitOne(const IOContext &ctx, IOControlBlock *cbp);
int IOGetEventsOne(const IOContext &ctx, IOEvent *event, TimeSpec *ts);

} // namespace blkio
} // namespace snurt
