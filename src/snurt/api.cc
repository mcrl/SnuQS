#include "api.h"
#include "context.h"
#include <memory>

namespace snurt {

static std::unique_ptr<Command> CreateCommand() {
  auto com = std::make_unique<Command>();

  if (com == nullptr) {
		throw std::runtime_error("Cannot create command");
  }

  return com;
}

int Init() {
  return context::Init();
}

void Deinit() {
  context::Deinit();
}

addr_t Malloc(size_t count) {
  addr_t addr;
  addr.ptr = malloc(count);
  return addr;
}

addr_t MallocAligned(size_t count, size_t align) {
  addr_t addr;
  addr.ptr = aligned_alloc(align, count);
  return addr;
}

int MemcpyH2S(CommandQueue &queue, addr_t dst, addr_t src, size_t count) {
  int retval = queue.RetriveAndResetRetval();
  if (retval) {
    return retval;
  }
  auto comm = CreateCommand();
  comm->opcode = CommandOpcode::kMemcpyH2S;
  comm->dst = dst;
  comm->src = src;
  comm->count = count;
  return queue.Enqueue(std::move(comm));
}

int MemcpyS2H(CommandQueue &queue, addr_t dst, addr_t src, size_t count) {
  int retval = queue.RetriveAndResetRetval();
  if (retval) {
    return retval;
  }
  auto comm = CreateCommand();
  comm->opcode = CommandOpcode::kMemcpyS2H;
  comm->dst = dst;
  comm->src = src;
  comm->count = count;
  return queue.Enqueue(std::move(comm));
}

int MemcpyH2D(CommandQueue &queue, addr_t dst, addr_t src, size_t count) {
  int retval = queue.RetriveAndResetRetval();
  if (retval) {
    return retval;
  }
  auto comm = CreateCommand();
  comm->opcode = CommandOpcode::kMemcpyH2D;
  comm->dst = dst;
  comm->src = src;
  comm->count = count;
  return queue.Enqueue(std::move(comm));
}

int MemcpyD2H(CommandQueue &queue, addr_t dst, addr_t src, size_t count) {
  int retval = queue.RetriveAndResetRetval();
  if (retval) {
    return retval;
  }
  auto comm = CreateCommand();
  comm->opcode = CommandOpcode::kMemcpyD2H;
  comm->dst = dst;
  comm->src = src;
  comm->count = count;
  return queue.Enqueue(std::move(comm));
}

int MemcpyS2D(CommandQueue &queue, addr_t dst, addr_t src, size_t count) {
  int retval = queue.RetriveAndResetRetval();
  if (retval) {
    return retval;
  }
  auto comm = CreateCommand();
  comm->opcode = CommandOpcode::kMemcpyS2D;
  comm->dst = dst;
  comm->src = src;
  comm->count = count;
  return queue.Enqueue(std::move(comm));
}

int MemcpyD2S(CommandQueue &queue, addr_t dst, addr_t src, size_t count) {
  int retval = queue.RetriveAndResetRetval();
  if (retval) {
    return retval;
  }
  auto comm = CreateCommand();
  comm->opcode = CommandOpcode::kMemcpyD2S;
  comm->dst = dst;
  comm->src = src;
  comm->count = count;
  return queue.Enqueue(std::move(comm));
}

int Synchronize(CommandQueue &queue) {
  int _retval = queue.RetriveAndResetRetval();
  if (_retval) {
    return _retval;
  }

  auto comm = CreateCommand();
  comm->opcode = CommandOpcode::kSynchronize;
  comm->count = 0;

  std::mutex mutex;
  std::condition_variable cv;
  comm->data = &cv;

  auto &data = comm->data;
  auto &status = comm->status;
  auto &retval = comm->retval;
  int err = queue.Enqueue(std::move(comm));
  if (err) return err;

  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [&] { return status == CommandStatus::kTerminated; });
  data = nullptr;
  cv.notify_one();

  int r = (retval == CommandRetval::kSuccess) ? 0 : -EINVAL;
  return r;
}

} // namespace snurt
