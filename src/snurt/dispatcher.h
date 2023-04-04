#pragma once
#include <vector>
#include <mutex>
#include <atomic>

#include "command_queue.h"

namespace snurt {

class Dispatcher {
  public:

  void RegisterQueue(CommandQueue *queue);
  int Init();
  void Deinit();
  void Loop();

  void Dispatch();

	private:
    std::mutex mutex_;
    std::vector<CommandQueue*> registered_queues_;
    std::atomic<bool> running_;
};

} // namespace snurt
