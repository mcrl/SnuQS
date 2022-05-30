#pragma once

#include <stddef.h>
#include <pthread.h>
#ifndef __cplusplus
# include <stdatomic.h>
#else
# include <atomic>
# define _Atomic(X) std::atomic< X >
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum socl_ops_t {
	H2S,
	S2H,
	H2D,
	D2H,
	SYNC,
	DSYNC,
	IOSYNC,
};

struct socl_command {
	enum socl_ops_t op;
	union {
		void *buf;
		void *daddr;
	};
	union {
		size_t io_addr;
		void *haddr;
	};
	size_t count;
	size_t tag;
	struct socl_command *next;
};

struct socl_command_queue {
	pthread_mutex_t lock;
	struct socl_command head;
	struct socl_command *tailp;
	_Atomic(int) tag;
};

typedef struct socl_command_queue* socl_command_queue_t;

int socl_init();
int socl_finalize();

int socl_create_command_queue(socl_command_queue_t *q);
int socl_destroy_command_queue(socl_command_queue_t q);

int socl_enqueueH2S(socl_command_queue_t q, void *buf, size_t io_addr, size_t count);
int socl_enqueueS2H(socl_command_queue_t q, void *buf, size_t io_addr, size_t count);
int socl_enqueueH2D(socl_command_queue_t q, void *daddr, void *haddr, size_t count);
int socl_enqueueD2H(socl_command_queue_t q, void *haddr, void *daddr, size_t count);
int socl_enqueueSync(socl_command_queue_t q);

#ifdef __cplusplus
}
#endif
