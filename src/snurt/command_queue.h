#pragma once
#include <list>
#include <mutex>
#include <stddef.h>
#include <condition_variable>
#include <memory>

#include "command.h"

namespace snurt {

enum class CommandQueueStatus {
  kStopped,
  kRunning,
};

class CommandQueue {
	public:
	CommandQueue();

	int Enqueue(std::unique_ptr<Command> &&comm);
	void Dequeue();
  void DequeueLocked();
	Command* Front();
	Command* FrontLocked();
	void Lock();
	void Unlock();
	CommandQueueStatus Status();

	void SetRetval(int retval);
	int RetriveAndResetRetval();

	bool empty();

	
	private:
	CommandQueueStatus status_;
	std::mutex mutex_;
	std::condition_variable cv_;

	std::list<std::unique_ptr<Command>> queue_;

	int retval_;

};

} // namespace snurt
