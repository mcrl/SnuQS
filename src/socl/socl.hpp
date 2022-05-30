#pragma once
#include <cstddef>

#include <atomic>
#include <queue>
#include "gpu_utils.hpp"

namespace socl {


enum Operation {
	H2S,
	S2H,
	H2D,
	D2H,
	SYNC,
	DSYNC,
	IOSYNC,
};

struct Command {
	Operation op;

	union {
		size_t from;
		void *from_p;
	};

	union {
		size_t to;
		void *to_p;
	};

	size_t count;

	Command(Operation op);
	Command(Operation op, size_t from, size_t to, size_t count);
	Command(Operation op, size_t from, void* to_p, size_t count);
	Command(Operation op, void* from_p, size_t to, size_t count);
	Command(Operation op, void* from_p, void* to_p, size_t count);
};


class CommandQueue {
public:
	int enqueueCommand(const Command &command);

//	std::queue<Command> commands_;
//	std::atomic<uint64_t> ncommand_;

	snuqs::gpu::stream_t stream;
};

using op_t = Operation;
using command_queue_t = CommandQueue*;

int initCommandQueue(command_queue_t q, size_t ncomm);
void deinitCommandQueue(command_queue_t q);
int enqueueH2S(command_queue_t q, void *buf, size_t io_addr, size_t count);
int enqueueS2H(command_queue_t q, void *buf, size_t io_addr, size_t count);
int enqueueH2D(command_queue_t q, void *daddr, void *haddr, size_t count);
int enqueueD2H(command_queue_t q, void *haddr, void *daddr, size_t count);
int enqueueIOSyncronize(command_queue_t q);
int enqueueSynchronize(command_queue_t q);

template<typename Function>
int enqueueOp(command_queue_t q, Function f, void *arg) {
	snuqs::gpu::launchHostFunc(q->stream, f, arg);
	return 0;
}


int enqueueRemoteH2S(command_queue_t q, void *buf, size_t node_no, size_t io_addr, size_t count);
int enqueueRemoteS2H(command_queue_t q, void *buf, size_t node_no, size_t local_io_addr, size_t count);

int main(int argc, char *argv[]);

};
