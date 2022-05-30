#pragma once

#include <cuda_runtime.h>

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


enum socl_gpu_op {
	SOCL_GPU_H2D,
	SOCL_GPU_D2H,
	SOCL_GPU_SYNC
};

struct socl_gpu_hdlr {
	int devno;
	cudaStream_t stream;
};

struct socl_gpu_command {
	enum socl_gpu_op op;
	void *dst;
	void *src;
	size_t count;

	int devno;
	cudaStream_t stream;
	struct socl_gpu_command *next;
};

typedef struct socl_gpu_hdlr* socl_gpu_hdlr_t;

int socl_gpu_init();
int socl_gpu_finalize();

int socl_gpu_hdlr_init(socl_gpu_hdlr_t *hdlr, int devno);
int socl_gpu_hdlr_deinit(socl_gpu_hdlr_t hdlr);

int socl_gpu_enqueueH2D(socl_gpu_hdlr_t hdlr, void *daddr, void *haddr, size_t count);
int socl_gpu_enqueueD2H(socl_gpu_hdlr_t hdlr, void *haddr, void *daddr, size_t count);
int socl_gpu_enqueueSync(socl_gpu_hdlr_t hdlr);

#ifdef __cplusplus
}
#endif
