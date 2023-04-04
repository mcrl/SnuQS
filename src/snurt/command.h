#pragma once

#include "addr.h"

namespace snurt {

enum class CommandStatus {
  kCreated = 0,
  kEnqueued,
  kRunning,
  kTerminated,
};

enum class CommandOpcode {
	kUnknown = 0,
	kMemcpyH2S,
	kMemcpyS2H,
	kMemcpyH2D,
	kMemcpyD2H,
	kMemcpyS2D,
	kMemcpyD2S,
	kSynchronize,
};

enum class CommandRetval {
	kSuccess = 0,
	kFailure,
};

struct Command {
	CommandStatus status = CommandStatus::kCreated;
	CommandOpcode opcode = CommandOpcode::kUnknown;
	CommandRetval retval = CommandRetval::kFailure;

  addr_t dst;
  addr_t src;

  size_t count;

  int dev_id;
  void *data;
};


} // namespace snurt
