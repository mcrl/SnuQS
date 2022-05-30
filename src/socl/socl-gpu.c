#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include "socl-gpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(EXIT_FAILURE);
	}
}

// TODO:
struct socl_gpu_request_queue {
	pthread_mutex_t lock;
	struct socl_gpu_command head;
	struct socl_gpu_command *tailp;
};

struct socl_gpu_completion_queue {
	pthread_mutex_t lock;
	struct socl_gpu_command head;
	struct socl_gpu_command *tailp;
};

struct socl_gpu_command_queue {
	int devno;
	struct socl_gpu_request_queue req_q;
	struct socl_gpu_completion_queue comp_q;
};

void* socl_gpu_loop(void *arg)
{
	//struct socl_gpu_command_queue *larg = (struct socl_gpu_command_queue*)arg;
	printf("Thread launched %d\n", sched_getcpu());
	return NULL;
}

pthread_t *threads;
struct socl_gpu_command_queue *queues;
static int ndev;

int socl_gpu_init()
{
	// TODO Launch d threads
	gpuErrchk(cudaGetDeviceCount(&ndev));

	threads = malloc(sizeof(pthread_t) * ndev);
	if (!threads) {
		fprintf(stderr, "Cannot allocate threads\n");
		exit(EXIT_FAILURE);
	}

	queues = malloc(sizeof(struct socl_gpu_command_queue) * ndev);
	if (!queues) {
		fprintf(stderr, "Cannot allocate thread queues\n");
		exit(EXIT_FAILURE);
	}

	int err;
	for (int i = 0; i < ndev; ++i) {
		pthread_mutex_init(&queues[i].req_q.lock, NULL);
		pthread_mutex_init(&queues[i].comp_q.lock, NULL);
		queues[i].devno = i;
		err = pthread_create(&threads[i], NULL, socl_gpu_loop, &queues[i]);
		if (err) return err;
	}
	return 0;
}
int socl_gpu_finalize()
{
	int err;
	for (int i = 0; i < ndev; ++i) {
		err = pthread_join(threads[i], NULL);
		if (err) return err;
		pthread_mutex_destroy(&queues[i].req_q.lock);
		pthread_mutex_destroy(&queues[i].comp_q.lock);
	}
	return 0;
}

int socl_gpu_hdlr_init(socl_gpu_hdlr_t *hdlr, int devno)
{
	socl_gpu_hdlr_t new_hdlr = malloc(sizeof(**hdlr));
	if (!new_hdlr) {
		fprintf(stderr, "Cannot allocate gpu hdlr\n");
		exit(EXIT_FAILURE);
	}

	new_hdlr->devno = devno;
	gpuErrchk(cudaSetDevice(devno));
	gpuErrchk(cudaStreamCreate(&new_hdlr->stream));

	*hdlr = new_hdlr;
	return 0;
}

int socl_gpu_hdlr_deinit(socl_gpu_hdlr_t hdlr)
{
	gpuErrchk(cudaStreamDestroy(hdlr->stream));
	free(hdlr);
	return 0;
}

struct socl_gpu_command* socl_gpu_alloc_command()
{
	struct socl_gpu_command *c = malloc(sizeof(struct socl_gpu_command));
	if (!c) {
		fprintf(stderr, "Cannot allocate gpu command\n");
		exit(EXIT_FAILURE);
	}
	return c;
}

int socl_gpu_enqueue_command(struct socl_gpu_command_queue *q, struct socl_gpu_command *c)
{
	struct socl_gpu_request_queue *req_q = &q->req_q;
	pthread_mutex_lock(&req_q->lock);

	req_q->tailp->next = c;
	req_q->tailp = c;

	pthread_mutex_unlock(&req_q->lock);
	return 0;
}

int socl_gpu_enqueueH2D(socl_gpu_hdlr_t hdlr, void *daddr, void *haddr, size_t count)
{
	struct socl_gpu_command *c = socl_gpu_alloc_command();
	c->op = SOCL_GPU_H2D;
	c->dst = daddr;
	c->src = haddr;
	c->count = count;
	c->devno = hdlr->devno;
	c->stream = hdlr->stream;

	return socl_gpu_enqueue_command(&queues[c->devno], c);
}

int socl_gpu_enqueueD2H(socl_gpu_hdlr_t hdlr, void *haddr, void *daddr, size_t count)
{
	struct socl_gpu_command *c = socl_gpu_alloc_command();
	c->op = SOCL_GPU_D2H;
	c->dst = haddr;
	c->src = daddr;
	c->count = count;
	c->devno = hdlr->devno;
	c->stream = hdlr->stream;

	return socl_gpu_enqueue_command(&queues[c->devno], c);
}

int socl_gpu_enqueueSync(socl_gpu_hdlr_t hdlr)
{
	struct socl_gpu_command *c = socl_gpu_alloc_command();
	c->op = SOCL_GPU_SYNC;
	return socl_gpu_enqueue_command(&queues[c->devno], c);
}
