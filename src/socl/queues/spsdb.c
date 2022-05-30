# include "spsdb.h"
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

int spsdb_init(struct spsdb *spsdb, size_t count) 
{
	spsdb->ents = malloc(sizeof(void*) * count);
	if (!spsdb->ents)
		return -ENOMEM;

	*(size_t*)(&spsdb->count) = count;

	atomic_init(&spsdb->head, 0);
	atomic_init(&spsdb->tail, 0);
	return 0;
}

void spsdb_deinit(struct spsdb *spsdb) 
{
	free(spsdb->ents);
}

int spsdb_enqueue(struct spsdb *spsdb, void *ent)
{
	if (atomic_load(&spsdb->tail) - atomic_load(&spsdb->head) >= spsdb->count) {
		return -ENOMEM;
	}
	uint64_t t = atomic_fetch_add(&spsdb->tail, 1);
	spsdb->ents[t % spsdb->count] = ent;
	//printf("enqueue %lu %lu\n", atomic_load(&spsdb->head), atomic_load(&spsdb->tail));
	return 0;
}

void* spsdb_dequeue(struct spsdb *spsdb)
{
	if (atomic_load(&spsdb->head) == atomic_load(&spsdb->tail)) {
		return NULL;
	}

	uint64_t h = atomic_fetch_add(&spsdb->head, 1);
	//printf("dequeue %lu %lu\n", atomic_load(&spsdb->head), atomic_load(&spsdb->tail));
	return spsdb->ents[h % spsdb->count];
}
