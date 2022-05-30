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



struct spsdb {
	void **ents;
	const size_t count;
	atomic_uint_fast64_t head;
	atomic_uint_fast64_t tail;
};


int spsdb_init(struct spsdb *spsdb, size_t count);
void spsdb_deinit(struct spsdb *spsdb);

int spsdb_enqueue(struct spsdb *spsdb, void *ent);
void* spsdb_dequeue(struct spsdb *spsdb);

#ifdef __cplusplus
}
#endif
