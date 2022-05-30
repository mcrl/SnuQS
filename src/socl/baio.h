#ifndef __BAIO_H__
#define __BAIO_H__

#include <pthread.h>
#include <liburing.h>
#ifdef __cplusplus
extern "C" {
#endif

#define BAIO_MAX_DEVS 128
#define BAIO_CONFIG_FILE "baio.config"

enum {
	BAIO_ASYNC_READ,
	BAIO_ASYNC_WRITE,
	BAIO_ASYNC_SYNC,
};

struct baio_device {
	struct io_uring rring;
	struct io_uring wring;
	unsigned num_reads;
	unsigned num_writes;

	char devname[32];
	int fd;
	size_t size_in_GB;
};

struct baio_handle {
	unsigned num_devices;
	unsigned queue_depth;
	size_t block_size;
	size_t block_size_per_device;
	struct baio_device *devices;
	size_t row_size;
	size_t width;
	size_t base;
};

//
// baio APIs
//
int baio_init(struct baio_handle *,  unsigned , const char *);
int baio_setup_blocks(struct baio_handle *, size_t , size_t , size_t, size_t);
void baio_finalize(struct baio_handle *hdlr);

void* baio_malloc(size_t size);

int baio_queue_read(struct baio_handle *hdlr, void *buf, off_t size, off_t offset);
int baio_queue_write(struct baio_handle *hdlr, void *buf, off_t size, off_t offset);
int baio_queue_submit(struct baio_handle *hdlr);
int baio_wait_all(struct baio_handle *hdlr);
int baio_fsync(struct baio_handle *hdlr);

#ifdef __cplusplus
};
#endif
#endif // __BAIO_H__
