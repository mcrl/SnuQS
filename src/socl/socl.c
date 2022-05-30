#include "socl.h"
#include "socl-gpu.h"
#include "socl-io.h"
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

int socl_init()
{
	int err;
//	err = socl_gpu_init();
//	if (err) return err;
	err = socl_io_init();
	if (err) return err;
	return 0;
}

int socl_finalize()
{
	int err;
//	err = socl_gpu_finalize();
//	if (err) return err;
	err = socl_io_finalize();
	if (err) return err;
	return 0;
}

static void print_command(struct socl_command *c)
{
	printf("---------------\n");
	printf("OP: %d\n", c->op);
	printf("buf1: 0x%lx\n", (size_t)c->buf);
	printf("buf2: 0x%lx\n", (size_t)c->haddr);
	printf("count: %lu\n", c->count);
	printf("---------------\n");
}

static void print_queue(socl_command_queue_t q)
{
	struct socl_command *c = q->head.next;
	if (!c) {
		printf("Empty\n");
	}
	while (c) {
		print_command(c);
		c = c->next;
	}
}

static int socl_enqueue_command(socl_command_queue_t q, struct socl_command *command)
{
	pthread_mutex_lock(&q->lock);

	q->tailp->next = command;
	q->tailp = command;

	pthread_mutex_unlock(&q->lock);

	return 0;
}

static struct socl_command* socl_dequeue_command(socl_command_queue_t q)
{
	pthread_mutex_lock(&q->lock);

	if (&q->head.next == NULL) {
		pthread_mutex_unlock(&q->lock);
		return NULL;
	}
	struct socl_command *command = q->head.next;
	q->head.next = command->next;
	if (q->head.next == NULL) {
		q->tailp = &q->head;
	}

	pthread_mutex_unlock(&q->lock);
	return command;
}

int socl_create_command_queue(socl_command_queue_t *q)
{
	socl_command_queue_t new_q;
	new_q = malloc(sizeof(**q));
	if (!new_q) {
		fprintf(stderr, "Cannot allocate memory\n");
		exit(EXIT_FAILURE);
	}
	pthread_mutex_init(&new_q->lock, NULL);

	new_q->head.next = NULL;
	new_q->tailp = &new_q->head;
	new_q->tag = 0;
	*q = new_q;
	return 0;
}

int socl_destroy_command_queue(socl_command_queue_t q)
{
	// TODO: Flush
	pthread_mutex_destroy(&q->lock);
	return 0;
}

static struct socl_command* socl_alloc_command(int tag)
{
	struct socl_command *c = malloc(sizeof(struct socl_command));
	if (!c) {
		fprintf(stderr, "Cannot allocate command\n");
		exit(EXIT_FAILURE);
	}
	return c;
}

int socl_enqueueH2S(socl_command_queue_t q, void *buf, size_t io_addr, size_t count)
{
	struct socl_command *c = socl_alloc_command(q->tag++);
	c->buf = buf;
	c->io_addr = io_addr;
	c->count = count;
	c->op = H2S;
	return socl_enqueue_command(q, c);
}

int socl_enqueueS2H(socl_command_queue_t q, void *buf, size_t io_addr, size_t count)
{
	struct socl_command *c = socl_alloc_command(q->tag++);
	c->buf = buf;
	c->io_addr = io_addr;
	c->count = count;
	c->op = S2H;
	return socl_enqueue_command(q, c);
}

int socl_enqueueH2D(socl_command_queue_t q, void *daddr, void *haddr, size_t count)
{
	struct socl_command *c = socl_alloc_command(q->tag++);
	c->daddr = daddr;
	c->haddr = haddr;
	c->count = count;
	c->op = H2D;
	return socl_enqueue_command(q, c);
}

int socl_enqueueD2H(socl_command_queue_t q, void *haddr, void *daddr, size_t count)
{
	struct socl_command *c = socl_alloc_command(q->tag++);
	c->daddr = daddr;
	c->haddr = haddr;
	c->count = count;
	c->op = D2H;
	return socl_enqueue_command(q, c);
}

int socl_enqueueSync(socl_command_queue_t q)
{
	struct socl_command *c = socl_alloc_command(q->tag++);
	c->op = SYNC;
	return socl_enqueue_command(q, c);
}
