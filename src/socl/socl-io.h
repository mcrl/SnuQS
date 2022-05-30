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


#include <stddef.h>

#ifdef _SOCL_IO_URING_SUPPORTED_

#include <liburing.h>
#define SOCL_IO_QUEUE_DEPTH 16

#else
#include <linux/aio_abi.h>
#define SOCL_IO_QUEUE_DEPTH 16
#endif

#define SOCL_IO_NAME_LENGTH 64
#define SOCL_IO_DEFAULT_BLOCK_SIZE (1ul << 12)
#define SOCL_IO_MAX_TRANSFER_SIZE 2147479552ul

enum socl_io_op {
	SOCL_IO_H2S,
	SOCL_IO_S2H,
	SOCL_IO_SYNC
};

struct socl_io_device {
	char name[SOCL_IO_NAME_LENGTH+1];
	int fd;

#ifdef _SOCL_IO_URING_SUPPORTED_
	struct io_uring ring;
#else
	aio_context_t ctx;
#endif
};

struct socl_io_hdlr {
	unsigned ndevs;
};

struct socl_io_command {
	enum socl_io_op op;
	void *buf;
	size_t addr;
	size_t count;

	struct socl_io_command *next;
};

typedef struct socl_io_hdlr* socl_io_hdlr_t;

int socl_io_init();
int socl_io_finalize();
int socl_io_set_block_size(size_t block_size);
int socl_io_hdlr_init(socl_io_hdlr_t *hdlr);
int socl_io_hdlr_deinit(socl_io_hdlr_t hdlr);
int socl_io_enqueueH2D(socl_io_hdlr_t hdlr, void *buf, size_t addr, size_t count);
int socl_io_enqueueD2H(socl_io_hdlr_t hdlr, void *buf, size_t addr, size_t count);
int socl_io_enqueueSync(socl_io_hdlr_t hdlr);

#ifdef __cplusplus
}
#endif
