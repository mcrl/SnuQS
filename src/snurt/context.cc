#include "context.h"

#include <cassert>
#include <vector>
#include <thread>
#include <atomic>

#include "dispatcher.h"

namespace snurt {
namespace context {

static std::vector<Dispatcher*> dispatchers;
static std::vector<std::thread> threads;

constexpr int num_dispatchers = 1;


static void DispatchLoop(Dispatcher *disp)
{
  disp->Loop();
}

int Init() {

  int err;
  err = ContextCUDAInit();
  if (err) return err;

  err = ContextIOInit();
  if (err) return err;

  for (int i = 0; i < num_dispatchers; ++i) {
    dispatchers.push_back(
        new Dispatcher()
        );
  }

  for (auto &dispatcher : dispatchers) {
    err = dispatcher->Init();
    if (err) return err;

    auto t = std::thread(DispatchLoop, dispatcher);
    t.detach();
    threads.push_back(std::move(t));
  }

  return 0;
}

void Deinit() {
  for (auto &dispatcher : dispatchers) {
    dispatcher->Deinit();
  }

  for (auto &dispatcher : dispatchers) {
    delete dispatcher;
  }

  ContextIODeinit();

  ContextCUDADeinit();
}

void RegisterQueue(CommandQueue *queue) {
  for (auto &dispatcher : dispatchers) {
    dispatcher->RegisterQueue(queue);
  }
}

} // namespace context
} // namespace snurt
