#include "command_queue.h"
#include "context.h"

namespace snurt {
	
CommandQueue::CommandQueue()
: status_(CommandQueueStatus::kRunning),
  retval_(0)
{
  context::RegisterQueue(this);
}

bool CommandQueue::empty() {
  return queue_.empty();
}

CommandQueueStatus CommandQueue::Status() {
  return status_;
}

int CommandQueue::Enqueue(std::unique_ptr<Command> &&comm) {
	std::lock_guard<std::mutex> guard(mutex_);
	if (Status() == CommandQueueStatus::kStopped)
	  return -EFAULT;
	comm->status = CommandStatus::kEnqueued;
	queue_.push_back(std::move(comm));

	return 0;
}

void CommandQueue::Dequeue() {
	std::lock_guard<std::mutex> guard(mutex_);
	if (queue_.size() == 0) {
		throw std::underflow_error("No command to dequeue");
	}

	queue_.pop_front();
}

void CommandQueue::DequeueLocked() {
	if (queue_.size() == 0) {
		throw std::underflow_error("No command to dequeue");
	}
	queue_.pop_front();
}


Command* CommandQueue::Front() {
	std::lock_guard<std::mutex> guard(mutex_);
	if (queue_.size() == 0) {
	  return nullptr;
	}

	return queue_.front().get();
}

Command* CommandQueue::FrontLocked() {
	if (queue_.size() == 0) {
	  return nullptr;
	}

	return queue_.front().get();
}

void CommandQueue::Lock() {
  mutex_.lock();
}

void CommandQueue::Unlock() {
  mutex_.unlock();
}

void CommandQueue::SetRetval(int retval) {
	std::lock_guard<std::mutex> guard(mutex_);
	retval_ = retval;
}

int CommandQueue::RetriveAndResetRetval() {
	std::lock_guard<std::mutex> guard(mutex_);
	int retval = retval_;
	retval_ = 0;
	return retval;
}
} // namespace snurt
