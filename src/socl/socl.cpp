#include "socl.hpp"
#include "gpu_utils.hpp"

namespace socl {

//
// Command
//

Command::Command(Operation op) 
: op(op) {
}

Command::Command(Operation op, size_t from, size_t to, size_t count)
: op(op), from(from), to(to), count(count) {
}

Command::Command(Operation op, size_t from, void* to_p, size_t count) 
: op(op), from(from), to_p(to_p), count(count) {
}

Command::Command(Operation op, void* from_p, size_t to, size_t count) 
: op(op), from_p(from_p), to(to), count(count) {
}

Command::Command(Operation op, void* from_p, void* to_p, size_t count) 
: op(op), from_p(from_p), to_p(to_p), count(count) {
}


//
// CommandQueue
//

int CommandQueue::enqueueCommand(const Command &command) {
	//commands_.push(command);
	return 0;
}

//
// API
//
int initCommandQueue(CommandQueue *q, size_t ncomm) {
	snuqs::gpu::streamCreate(&q->stream);
	return 0;
}

void deinitCommandQueue(CommandQueue *q) {
	snuqs::gpu::streamDestroy(q->stream);
}

int enqueueH2S(CommandQueue *q, void *buf, size_t io_addr, size_t count) {
	//q->enqueueCommand(Command(H2S, buf, io_addr, count));
	return 0;
}

int enqueueS2H(CommandQueue *q, void *buf, size_t io_addr, size_t count) {
	//q->enqueueCommand(Command(S2H, io_addr, buf, count));
	return 0;
}

int enqueueH2D(CommandQueue *q, void *daddr, void *haddr, size_t count) {
    //q->enqueueCommand(Command(H2D, haddr, daddr, count));
	snuqs::gpu::MemcpyAsyncH2D(daddr, haddr, count, q->stream);
	return 0;
}

int enqueueD2H(CommandQueue *q, void *haddr, void *daddr, size_t count) {
	//q->enqueueCommand(Command(D2H, daddr, haddr, count));
	snuqs::gpu::MemcpyAsyncD2H(haddr, daddr, count, q->stream);
	return 0;
}

int enqueueIOSyncronize(CommandQueue *q) {
	//q->enqueueCommand(Command(IOSYNC));
	return 0;
}

int enqueueSynchronize(CommandQueue *q) {
	snuqs::gpu::streamSynchronize(q->stream);
	//q->enqueueCommand(Command(SYNC));
	return 0;
}






/*
int enqueueRemoteH2S(CommandQueue *q, void *buf, size_t node_no, size_t io_addr, size_t count) {
	return 0;
}

int enqueueRemoteS2H(CommandQueue *q, void *buf, size_t node_no, size_t local_io_addr, size_t count) {
	return 0;
}
*/

} // namespace socl
