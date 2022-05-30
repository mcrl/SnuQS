#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __cplusplus
#include <stdatomic.h>
#else
#define _Atomic(X) std::atomic< X >
#include <atomic>
#endif


struct spsdu_node {
	void *ent;
	atomic_uintptr_t next;
};

struct spsdu {
	atomic_uintptr_t head;
	atomic_uintptr_t tail;
};


int spsdu_init(struct spsdu *spsdu);
void spsdu_deinit(struct spsdu *spsdu);

int spsdu_enqueue(struct spsdu *spsdu, void *ent);
void* spsdu_dequeue(struct spsdu *spsdu);

#ifdef __cplusplus
}
#endif
